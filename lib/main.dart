import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_ffi_uvc/flutter_ffi_uvc.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const X5Page(),
    ),
  );
}

class X5Page extends StatefulWidget {
  const X5Page({super.key});

  @override
  State<X5Page> createState() => _X5PageState();
}

class _X5PageState extends State<X5Page> with SingleTickerProviderStateMixin {
  final List<String> _lines = [];
  final ScrollController _scroll = ScrollController();

  int? _textureId;
  int _w = 16;
  int _h = 9;
  bool _busy = false;
  bool _streaming = false;
  bool _view360 = true; // true = shader 360 view, false = flat GPU texture

  // Realtime 360 rendering
  final ValueNotifier<ui.Image?> _frameImg = ValueNotifier(null);
  final ValueNotifier<int> _viewTick = ValueNotifier(0);
  ui.FragmentShader? _shader;
  late final Ticker _ticker;
  bool _grabbing = false;
  double _yaw = 0, _pitch = 0, _fov = 1.2; // radians
  double _fovAtScaleStart = 1.2;
  int _lensMode = 0; // 0 = equirectangular, 1 = dual fisheye

  // FPS counter (updates once a second, no per-frame setState)
  final ValueNotifier<String> _fpsText = ValueNotifier('');
  int _frameCount = 0;
  DateTime _fpsStamp = DateTime.now();

  StreamSubscription<UvcStreamError>? _errSub;
  StreamSubscription<UvcDeviceEvent>? _devSub;
  StreamSubscription<UvcStallEvent>? _stallSub;

  void _log(String s) {
    debugPrint('X5 $s');
    if (!mounted) return;
    setState(() {
      _lines.add(s);
      if (_lines.length > 300) _lines.removeAt(0);
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  @override
  void initState() {
    super.initState();
    _ticker = createTicker((_) => _grab());

    ui.FragmentProgram.fromAsset('shaders/pano.frag')
        .then((p) {
          if (!mounted) return;
          setState(() => _shader = p.fragmentShader());
        })
        .catchError((e) {
          _log('Shader load failed: $e');
        });

    _errSub = uvcCamera.streamErrors.listen(
      (UvcStreamError e) => _log('STREAM ERROR: ${e.message}'),
    );
    _devSub = uvcCamera.deviceEvents.listen(
      (UvcDeviceEvent e) => _log('USB event: ${e.type}'),
    );
  }

  // Frames wider than this are downscaled (on a background thread) before
  // the GPU upload. Cuts upload size a lot at 2880x1440 with little visible loss.
  static const int _maxUploadW = 2880;

  int _inFlight = 0; // up to 2 frames decode in parallel
  int _seqIssued = 0;
  int _seqShown = 0;
  int _copyUsAcc = 0;
  int _decodeUsAcc = 0;
  int _srcW = 0, _srcH = 0;

  // Called every vsync while streaming in 360 mode.
  void _grab() {
    if (!_view360 || _inFlight >= 2) return;
    final sw = Stopwatch()..start();
    final frame = uvcCamera.copyLatestFrame();
    if (frame == null) return;
    final copyUs = sw.elapsedMicroseconds;
    _inFlight++;
    _decode(frame.rgbaBytes, frame.width, frame.height, ++_seqIssued, copyUs);
  }

  Future<void> _decode(
    Uint8List bytes,
    int w,
    int h,
    int seq,
    int copyUs,
  ) async {
    final sw = Stopwatch()..start();
    ui.Image img;
    try {
      final buf = await ui.ImmutableBuffer.fromUint8List(bytes);
      final desc = ui.ImageDescriptor.raw(
        buf,
        width: w,
        height: h,
        pixelFormat: ui.PixelFormat.rgba8888,
      );
      final down = w > _maxUploadW;
      final codec = await desc.instantiateCodec(
        targetWidth: down ? _maxUploadW : null,
        targetHeight: down ? (h * _maxUploadW / w).round() : null,
      );
      final fi = await codec.getNextFrame();
      img = fi.image;
      codec.dispose();
      desc.dispose();
      buf.dispose();
    } catch (e) {
      _inFlight--;
      _log('decode error: $e');
      return;
    }
    _inFlight--;

    // drop frames that finished out of order or after dispose
    if (!mounted || seq < _seqShown) {
      img.dispose();
      return;
    }
    _seqShown = seq;

    final old = _frameImg.value;
    _frameImg.value = img; // only the painter repaints, no setState
    WidgetsBinding.instance.addPostFrameCallback((_) => old?.dispose());

    _srcW = w;
    _srcH = h;
    _copyUsAcc += copyUs;
    _decodeUsAcc += sw.elapsedMicroseconds;
    _frameCount++;
    final now = DateTime.now();
    final ms = now.difference(_fpsStamp).inMilliseconds;
    if (ms >= 1000) {
      final n = _frameCount;
      _fpsText.value =
          '${_srcW}x$_srcH  ${(n * 1000 / ms).toStringAsFixed(1)} fps  '
          'copy ${(_copyUsAcc / n / 1000).toStringAsFixed(1)}ms  '
          'decode ${(_decodeUsAcc / n / 1000).toStringAsFixed(1)}ms';
      _frameCount = 0;
      _copyUsAcc = 0;
      _decodeUsAcc = 0;
      _fpsStamp = now;
    }
  }

  Future<void> _start() async {
    if (_busy || _streaming) return;
    setState(() => _busy = true);
    try {
      uvcCamera.setLogLevel(UvcLogLevel.debug);

      final camOk = await uvcCamera.ensureCameraPermission();
      _log('CAMERA permission: $camOk');

      final devices = await uvcCamera.listUsbDevices();
      _log('UVC devices found: ${devices.length}');
      if (devices.isEmpty) {
        _log('No UVC device. Plug in the X5 (Webcam Mode) and tap Start.');
        return;
      }

      try {
        await uvcCamera.openUsbDevice(devices.first.deviceId);
      } catch (e) {
        _log('openUsbDevice failed: $e (${uvcCamera.lastError})');
        return;
      }
      _log('Device opened');

      final modes = await Future.value(uvcCamera.supportedModes());
      for (final m in modes) {
        _log('MODE ${m.label}');
      }

      final tex = await uvcCamera.createPreviewTexture();
      final res = await uvcCamera.startPreviewAuto(
        preference: UvcAutoPreviewPreference.quality,
      );
      for (final a in res.attempts) {
        _log(
          'try ${a.mode.label}: ${a.success ? "OK" : "failed"} ${a.lastError}',
        );
      }
      if (!res.success || res.mode == null) {
        _log('startPreviewAuto: no mode streamed frames');
        await uvcCamera.disposePreviewTexture(tex);
        return;
      }

      final mode = res.mode!;
      await uvcCamera.attachPreviewTexture(
        tex,
        width: mode.width,
        height: mode.height,
      );
      _log('Preview running: ${mode.label}');
      setState(() {
        _textureId = tex;
        _w = mode.width;
        _h = mode.height;
        _streaming = true;
      });

      uvcCamera.enableStallDetection(
        const UvcStallDetectionConfig(
          stallTimeout: Duration(seconds: 2),
          autoRestart: false,
          maxRestartAttempts: 3,
        ),
      );
      _stallSub = uvcCamera.stallEvents.listen(
        (UvcStallEvent e) => _log('STALL event: ${e.type}'),
      );

      _frameCount = 0;
      _fpsStamp = DateTime.now();
      if (!_ticker.isActive) _ticker.start();
    } catch (e) {
      _log('ERROR: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _stop() async {
    if (_ticker.isActive) _ticker.stop();
    await _stallSub?.cancel();
    _stallSub = null;
    try {
      uvcCamera.disableStallDetection();
    } catch (_) {}
    try {
      uvcCamera.stopPreview();
    } catch (_) {}
    final tex = _textureId;
    if (tex != null) {
      try {
        await uvcCamera.disposePreviewTexture(tex);
      } catch (_) {}
    }
    try {
      await uvcCamera.closeUsbDevice();
    } catch (_) {}
    if (mounted) {
      setState(() {
        _textureId = null;
        _streaming = false;
      });
    }
    _log('Stopped');
  }

  @override
  void dispose() {
    _errSub?.cancel();
    _devSub?.cancel();
    _ticker.dispose();
    _stop();
    _frameImg.value?.dispose();
    _frameImg.dispose();
    _viewTick.dispose();
    _fpsText.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Widget _buildPreview() {
    if (!_streaming) {
      return const Center(child: Text('Press Start'));
    }

    // Flat equirectangular view straight from the GPU texture. No copies.
    if (!_view360) {
      if (_textureId == null) return const SizedBox.shrink();
      return Center(
        child: AspectRatio(
          aspectRatio: _w / _h,
          child: Texture(textureId: _textureId!),
        ),
      );
    }

    // Realtime 360 perspective view via fragment shader.
    if (_shader == null) {
      return const Center(child: Text('Loading shader'));
    }
    return GestureDetector(
      onScaleStart: (_) => _fovAtScaleStart = _fov,
      onScaleUpdate: (d) {
        _yaw -= d.focalPointDelta.dx * 0.005 * _fov;
        _pitch = (_pitch - d.focalPointDelta.dy * 0.005 * _fov).clamp(
          -1.5,
          1.5,
        );
        if (d.scale != 1.0) {
          _fov = (_fovAtScaleStart / d.scale).clamp(0.5, 2.2);
        }
        _viewTick.value++;
      },
      child: CustomPaint(
        size: Size.infinite,
        painter: PanoPainter(
          _shader!,
          _frameImg,
          () => _yaw,
          () => _pitch,
          () => _fov,
          () => _lensMode.toDouble(),
          Listenable.merge([_frameImg, _viewTick]),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('X5 UVC test')),
      body: Column(
        children: [
          SizedBox(
            height: 300,
            width: double.infinity,
            child: Stack(
              children: [
                Positioned.fill(
                  child: ColoredBox(
                    color: const Color(0xFF0B0D0F),
                    child: _buildPreview(),
                  ),
                ),
                Positioned(
                  left: 8,
                  top: 8,
                  child: ValueListenableBuilder<String>(
                    valueListenable: _fpsText,
                    builder: (_, t, __) => Text(
                      t,
                      style: const TextStyle(
                        fontSize: 11,
                        color: Colors.greenAccent,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 8,
              children: [
                FilledButton(
                  onPressed: (_busy || _streaming) ? null : _start,
                  child: const Text('Start'),
                ),
                OutlinedButton(
                  onPressed: _streaming ? _stop : null,
                  child: const Text('Stop'),
                ),
                const Text('360'),
                Switch(
                  value: _view360,
                  onChanged: (v) => setState(() => _view360 = v),
                ),
                SegmentedButton<int>(
                  showSelectedIcon: false,
                  segments: const [
                    ButtonSegment(value: 0, label: Text('Equirect')),
                    ButtonSegment(value: 1, label: Text('Fisheye')),
                  ],
                  selected: {_lensMode},
                  onSelectionChanged: (v) {
                    setState(() => _lensMode = v.first);
                    _viewTick.value++;
                  },
                ),
              ],
            ),
          ),
          Expanded(
            child: Container(
              color: const Color(0xFF14171A),
              width: double.infinity,
              child: SelectionArea(
                child: ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.all(8),
                  itemCount: _lines.length,
                  itemBuilder: (_, i) => Text(
                    _lines[i],
                    style: const TextStyle(
                      fontSize: 11,
                      color: Color(0xFFB4C0C3),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class PanoPainter extends CustomPainter {
  PanoPainter(
    this.shader,
    this.img,
    this.yaw,
    this.pitch,
    this.fov,
    this.mode,
    Listenable repaint,
  ) : super(repaint: repaint);

  final ui.FragmentShader shader;
  final ValueNotifier<ui.Image?> img;
  final double Function() yaw;
  final double Function() pitch;
  final double Function() fov;
  final double Function() mode;

  @override
  void paint(Canvas canvas, Size size) {
    final i = img.value;
    if (i == null) return;
    shader
      ..setFloat(0, size.width)
      ..setFloat(1, size.height)
      ..setFloat(2, yaw())
      ..setFloat(3, pitch())
      ..setFloat(4, fov())
      ..setFloat(5, mode())
      ..setFloat(6, 3.4906585) // 200 deg lens
      ..setImageSampler(0, i);
    canvas.drawRect(Offset.zero & size, Paint()..shader = shader);
  }

  @override
  bool shouldRepaint(covariant PanoPainter old) => false; // driven by `repaint`
}
