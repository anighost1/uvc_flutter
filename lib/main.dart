import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_ffi_uvc/flutter_ffi_uvc.dart';

import 'dart:typed_data';
import 'package:panorama_viewer/panorama_viewer.dart';

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

class _X5PageState extends State<X5Page> {
  final List<String> _lines = [];
  final ScrollController _scroll = ScrollController();

  int? _textureId;
  int _w = 16;
  int _h = 9;
  bool _busy = false;
  bool _streaming = false;
  bool _quality = false; // true = try the largest mode (2880x1440) first
  ui.Image? _snap;
  Uint8List? _jpgBytes;

  Timer? _poll;
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
    _errSub = uvcCamera.streamErrors.listen(
      (UvcStreamError e) => _log('STREAM ERROR: ${e.message}'),
    );
    _devSub = uvcCamera.deviceEvents.listen(
      (UvcDeviceEvent e) => _log('USB event: ${e.type}'),
    );
  }

  // Looks at the raw frame: average brightness + a snapshot to draw
  void _inspectFrame() {
    final frame = uvcCamera.copyLatestFrame();
    if (frame == null) {
      _log('frame check: no frame yet');
      return;
    }
    final b = frame.rgbaBytes;
    int sum = 0;
    int n = 0;
    int maxV = 0;
    for (int i = 0; i + 2 < b.length; i += 4 * 997) {
      final v = (b[i] + b[i + 1] + b[i + 2]) ~/ 3;
      sum += v;
      n++;
      if (v > maxV) maxV = v;
    }
    final avg = n == 0 ? 0 : sum / n;
    _log(
      'frame ${frame.width}x${frame.height}  brightness avg ${avg.toStringAsFixed(1)} / max $maxV (0-255)',
    );

    ui.decodeImageFromPixels(
      b,
      frame.width,
      frame.height,
      ui.PixelFormat.rgba8888,
      (ui.Image img) async {
        final byteData = await img.toByteData(format: ui.ImageByteFormat.png);

        if (!mounted || byteData == null) return;

        setState(() {
          _snap = img;
          _jpgBytes = byteData.buffer.asUint8List();
        });
      },
    );
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
        preference: _quality
            ? UvcAutoPreviewPreference.quality
            : UvcAutoPreviewPreference.reliability,
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

      _poll = Timer.periodic(
        const Duration(milliseconds: 100),
        (_) => _inspectFrame(),
      );
    } catch (e) {
      _log('ERROR: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _stop() async {
    _poll?.cancel();
    _poll = null;
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
        _snap = null;
      });
    }
    _log('Stopped');
  }

  @override
  void dispose() {
    _errSub?.cancel();
    _devSub?.cancel();
    _stop();
    _scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('X5 UVC test')),
      body: Column(
        children: [
          // 1) The Texture (what the plugin draws directly)
          SizedBox(
            height: 280,
            child: _jpgBytes == null
                ? const Center(child: Text('Waiting for frames'))
                : PanoramaViewer(
                    minZoom: 0.5,
                    maxZoom: 5,
                    animSpeed: 0.0,
                    child: Image.memory(_jpgBytes!, gaplessPlayback: true),
                  ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Row(
              children: [
                FilledButton(
                  onPressed: (_busy || _streaming) ? null : _start,
                  child: const Text('Start'),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _streaming ? _stop : null,
                  child: const Text('Stop'),
                ),
                const Spacer(),
                const Text('Max quality'),
                Switch(
                  value: _quality,
                  onChanged: _streaming
                      ? null
                      : (v) => setState(() => _quality = v),
                ),
              ],
            ),
          ),
          // 2) A snapshot of the raw frame, drawn by Flutter itself
          Container(
            height: 120,
            width: double.infinity,
            color: const Color(0xFF0B0D0F),
            child: _snap == null
                ? const Center(child: Text('Snapshot appears here'))
                : RawImage(image: _snap, fit: BoxFit.contain),
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
