import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_ffi_uvc/flutter_ffi_uvc.dart';

import 'detection.dart';

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

class _X5PageState extends State<X5Page>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  // ---- log pane (rebuilds only itself, never the whole page) ----
  final List<String> _lines = [];
  final ValueNotifier<int> _logTick = ValueNotifier(0);
  final ScrollController _scroll = ScrollController();
  bool _disposed = false;

  // ---- camera state ----
  int? _textureId;
  int _w = 16;
  int _h = 9;
  bool _busy = false;
  bool _streaming = false;
  bool _view360 = true; // true = shader 360 view, false = flat GPU texture

  // ---- realtime 360 rendering ----
  final ValueNotifier<ui.Image?> _frameImg = ValueNotifier(null);
  final ValueNotifier<int> _viewTick = ValueNotifier(0);
  ui.FragmentShader? _shader;
  late final Ticker _ticker;
  double _yaw = 0, _pitch = 0, _fov = 1.2; // radians
  double _fovAtScaleStart = 1.2;

  // ---- frame pipeline ----
  static const int _maxUploadW = 2880; // frames wider than this get downscaled
  static const int _maxInFlight = 2; // frames decoding/uploading in parallel

  final Stopwatch _clock = Stopwatch()..start();
  int _nextGrabMs = 0;
  int _dupStreak = 0;
  int _inFlight = 0;
  int _seqIssued = 0;
  int _seqShown = 0;
  int _lastSig = 0;

  // ---- stats (updated once a second by a timer, not per frame) ----
  final ValueNotifier<String> _fpsText = ValueNotifier('');
  Timer? _statsTimer;
  DateTime _fpsStamp = DateTime.now();
  int _shownCount = 0;
  int _grabs = 0;
  int _srcUnique = 0;
  int _copyUsAcc = 0;
  int _decodeUsAcc = 0;
  int _srcW = 0, _srcH = 0;

  // ---- detection API ----
  final DetectionClient _api = DetectionClient();
  final ValueNotifier<List<Detection>> _dets = ValueNotifier(const []);
  bool _apiOn = true;
  bool _apiBusy = false;
  int _nextApiMs = 0;
  int _detAtMs = 0;
  int _apiFailStreak = 0;
  int _apiOk = 0;
  int _apiMsAcc = 0;
  int _apiErr = 0;
  String _apiLastErr = '';

  // ---- stream errors are counted, then logged once a second ----
  int _errCount = 0;
  String _lastErr = '';

  StreamSubscription<UvcStreamError>? _errSub;
  StreamSubscription<UvcDeviceEvent>? _devSub;
  StreamSubscription<UvcStallEvent>? _stallSub;

  void _log(String s) {
    debugPrint('X5 $s');
    if (_disposed) return;
    _lines.add(s);
    if (_lines.length > 200) _lines.removeAt(0);
    _logTick.value++;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_disposed && _scroll.hasClients) {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _ticker = createTicker((_) => _grab());
    _api.onLog = _log;
    _api.connect();

    ui.FragmentProgram.fromAsset('shaders/pano.frag')
        .then((p) {
          if (!mounted) return;
          setState(() => _shader = p.fragmentShader());
        })
        .catchError((e) {
          _log('Shader load failed: $e');
        });

    _errSub = uvcCamera.streamErrors.listen((UvcStreamError e) {
      _errCount++;
      _lastErr = e.message;
    });
    _devSub = uvcCamera.deviceEvents.listen(
      (UvcDeviceEvent e) => _log('USB event: ${e.type}'),
    );
    _statsTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _statsTick(),
    );
  }

  // Stop all per-frame work while the app is in the background.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      if (_streaming && !_ticker.isActive) _ticker.start();
    } else {
      if (_ticker.isActive) _ticker.stop();
    }
  }

  // ------------------------------------------------------------------
  // Frame pipeline
  // ------------------------------------------------------------------

  // Called every vsync while streaming. Does nothing unless it is time to poll.
  void _grab() {
    if (!_view360 || _inFlight >= _maxInFlight) return;
    final nowMs = _clock.elapsedMilliseconds;
    if (nowMs < _nextGrabMs) return;

    final sw = Stopwatch()..start();
    final frame = uvcCamera.copyLatestFrame();
    if (frame == null) {
      _nextGrabMs = nowMs + 100;
      return;
    }
    final bytes = frame.rgbaBytes;
    _copyUsAcc += sw.elapsedMicroseconds;
    _grabs++;

    // Cheap signature of ~2k sampled bytes: skip pictures we already showed.
    int sig = 17;
    final step = ((bytes.length ~/ 2048) & ~3) + 5;
    for (int i = 0; i < bytes.length; i += step) {
      sig = (sig * 31 + bytes[i]) & 0x3fffffff;
    }
    if (sig == _lastSig) {
      // No new frame yet: poll again soon, and back off if it stays quiet.
      _dupStreak++;
      _nextGrabMs = _clock.elapsedMilliseconds + (_dupStreak > 3 ? 40 : 12);
      return;
    }
    _dupStreak = 0;
    _lastSig = sig;
    _nextGrabMs = _clock.elapsedMilliseconds + 10;
    _srcUnique++;
    _srcW = frame.width;
    _srcH = frame.height;

    _inFlight++;
    _decode(bytes, frame.width, frame.height, ++_seqIssued);
  }

  Future<void> _decode(Uint8List bytes, int w, int h, int seq) async {
    final sw = Stopwatch()..start();

    // When it is time to ask the API, also make a small copy of this frame.
    final wantApi =
        _apiOn &&
        _api.ready &&
        !_apiBusy &&
        _clock.elapsedMilliseconds >= _nextApiMs;
    if (wantApi) _apiBusy = true;

    ui.Image img;
    ui.Image? small;
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
      img = (await codec.getNextFrame()).image;
      codec.dispose();

      if (wantApi) {
        final sendW = ApiConfig.sendWidth < w ? ApiConfig.sendWidth : w;
        final sendH = (h * sendW / w).round();
        final c2 = await desc.instantiateCodec(
          targetWidth: sendW,
          targetHeight: sendH,
        );
        small = (await c2.getNextFrame()).image;
        c2.dispose();
      }
      desc.dispose();
      buf.dispose();
    } catch (e) {
      _inFlight--;
      if (wantApi) _apiBusy = false;
      _log('decode error: $e');
      return;
    }
    _inFlight--;

    // drop frames that finished out of order or after dispose
    if (_disposed || seq < _seqShown) {
      img.dispose();
      small?.dispose();
      if (wantApi) _apiBusy = false;
      return;
    }
    _seqShown = seq;

    final old = _frameImg.value;
    _frameImg.value = img; // only the painter repaints, no setState
    WidgetsBinding.instance.addPostFrameCallback((_) => old?.dispose());

    _decodeUsAcc += sw.elapsedMicroseconds;
    _shownCount++;

    if (small != null) unawaited(_detect(small));
  }

  // Encode the small frame, send it to the API, store the outlines.
  Future<void> _detect(ui.Image small) async {
    final sw = Stopwatch()..start();
    bool failed = false;
    try {
      final w = small.width, h = small.height;
      ByteData? bd;
      try {
        bd = await small.toByteData(format: ui.ImageByteFormat.rawRgba);
      } finally {
        small.dispose();
      }
      if (bd == null) throw 'could not read frame pixels';
      final rgba = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);

      final dataUrl = await encodeJpegDataUrl(rgba, w, h);
      final result = await _api.detect(dataUrl, w, h);
      if (_disposed) return;

      _dets.value = result;
      _detAtMs = _clock.elapsedMilliseconds;
      _apiOk++;
      _apiMsAcc += sw.elapsedMilliseconds;
      _apiFailStreak = 0;
    } catch (e) {
      failed = true;
      _apiErr++;
      _apiLastErr = '$e';
      _apiFailStreak++;
    } finally {
      _apiBusy = false;
      final wait = failed
          ? (250 * _apiFailStreak).clamp(250, 2000).toInt()
          : ApiConfig.minIntervalMs;
      _nextApiMs = _clock.elapsedMilliseconds + wait;
    }
  }

  // Runs once a second.
  void _statsTick() {
    if (_errCount > 0) {
      _log('STREAM ERROR x$_errCount: $_lastErr');
      _errCount = 0;
    }
    if (_apiErr > 0) {
      _log('API ERROR x$_apiErr: $_apiLastErr');
      _apiErr = 0;
    }
    // remove outlines that have not been refreshed
    if (_dets.value.isNotEmpty &&
        _clock.elapsedMilliseconds - _detAtMs > ApiConfig.resultTtlMs) {
      _dets.value = const [];
    }
    if (!_streaming || !_view360) {
      if (_fpsText.value.isNotEmpty) _fpsText.value = '';
      return;
    }
    final now = DateTime.now();
    final ms = now.difference(_fpsStamp).inMilliseconds;
    if (ms <= 0) return;
    final copyMs = _grabs == 0 ? 0.0 : _copyUsAcc / _grabs / 1000;
    final decMs = _shownCount == 0 ? 0.0 : _decodeUsAcc / _shownCount / 1000;
    final apiMs = _apiOk == 0 ? 0 : (_apiMsAcc / _apiOk).round();
    _fpsText.value =
        '${_srcW}x$_srcH  '
        'cam ${(_srcUnique * 1000 / ms).toStringAsFixed(1)}  '
        'show ${(_shownCount * 1000 / ms).toStringAsFixed(1)} fps  '
        'copy ${copyMs.toStringAsFixed(0)}ms  '
        'decode ${decMs.toStringAsFixed(0)}ms'
        '${_apiOn ? '\napi ${(_apiOk * 1000 / ms).toStringAsFixed(1)}/s ${apiMs}ms  ${_dets.value.length} objects' : ''}';
    _shownCount = 0;
    _srcUnique = 0;
    _grabs = 0;
    _copyUsAcc = 0;
    _decodeUsAcc = 0;
    _apiOk = 0;
    _apiMsAcc = 0;
    _fpsStamp = now;
  }

  // ------------------------------------------------------------------
  // Camera control
  // ------------------------------------------------------------------

  Future<void> _start() async {
    if (_busy || _streaming) return;
    setState(() => _busy = true);
    try {
      if (!kReleaseMode) uvcCamera.setLogLevel(UvcLogLevel.debug);

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

      _fpsStamp = DateTime.now();
      _shownCount = 0;
      _srcUnique = 0;
      _grabs = 0;
      _copyUsAcc = 0;
      _decodeUsAcc = 0;
      _dupStreak = 0;
      _nextGrabMs = 0;
      _nextApiMs = 0;
      _apiFailStreak = 0;
      if (!_ticker.isActive) _ticker.start();
    } catch (e) {
      _log('ERROR: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _stop() async {
    if (_ticker.isActive) _ticker.stop();
    if (!_disposed) _dets.value = const [];
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
    _disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _statsTimer?.cancel();
    _errSub?.cancel();
    _devSub?.cancel();
    _stop(); // its synchronous part stops the ticker
    _ticker.dispose();
    _frameImg.value?.dispose();
    _frameImg.dispose();
    _viewTick.dispose();
    _api.close();
    _dets.dispose();
    _fpsText.dispose();
    _logTick.dispose();
    _scroll.dispose();
    super.dispose();
  }

  // ------------------------------------------------------------------
  // UI
  // ------------------------------------------------------------------

  Widget _buildPreview() {
    if (!_streaming) {
      return const Center(child: Text('Press Start'));
    }

    // Flat view straight from the GPU texture. No copies at all.
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
      child: Stack(
        fit: StackFit.expand,
        children: [
          RepaintBoundary(
            child: CustomPaint(
              size: Size.infinite,
              painter: PanoPainter(
                _shader!,
                _frameImg,
                () => _yaw,
                () => _pitch,
                () => _fov,
                Listenable.merge([_frameImg, _viewTick]),
              ),
            ),
          ),
          IgnorePointer(
            child: RepaintBoundary(
              child: CustomPaint(
                size: Size.infinite,
                painter: OutlinePainter(
                  _dets,
                  () => _yaw,
                  () => _pitch,
                  () => _fov,
                  Listenable.merge([_dets, _viewTick]),
                ),
              ),
            ),
          ),
        ],
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
                const Text('Detect'),
                Switch(
                  value: _apiOn,
                  onChanged: (v) {
                    setState(() => _apiOn = v);
                    if (!v) _dets.value = const [];
                  },
                ),
                ValueListenableBuilder<bool>(
                  valueListenable: _api.connected,
                  builder: (_, c, __) => Text(
                    c ? 'Server: connected' : 'Server: offline',
                    style: TextStyle(
                      fontSize: 12,
                      color: c ? Colors.greenAccent : Colors.redAccent,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: Container(
              color: const Color(0xFF14171A),
              width: double.infinity,
              child: SelectionArea(
                child: ValueListenableBuilder<int>(
                  valueListenable: _logTick,
                  builder: (_, __, ___) => ListView.builder(
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
    Listenable repaint,
  ) : super(repaint: repaint);

  final ui.FragmentShader shader;
  final ValueNotifier<ui.Image?> img;
  final double Function() yaw;
  final double Function() pitch;
  final double Function() fov;
  final Paint _paint = Paint();

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
      ..setImageSampler(0, i);
    _paint.shader = shader;
    canvas.drawRect(Offset.zero & size, _paint);
  }

  @override
  bool shouldRepaint(covariant PanoPainter old) => false; // driven by `repaint`
}
