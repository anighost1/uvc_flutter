import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:socket_io_client/socket_io_client.dart' as io;

// =====================================================================
// CONFIG (same server and events as the web version)
// =====================================================================

class ApiConfig {
  /// Same server as NEXT_PUBLIC_YOLO_SERVER_URL in the web app.
  /// On a phone "localhost" is the phone itself, so use your computer's LAN IP.
  /// Override at build time:
  ///   flutter run --dart-define=YOLO_SERVER_URL=http://192.168.1.20:3000
  static const String serverUrl = String.fromEnvironment(
    'YOLO_SERVER_URL',
    defaultValue: 'http://192.168.0.111:8000',
  );

  static const String sendEvent = 'detect-frame';
  static const String resultEvent = 'detection-result';
  static const String errorEvent = 'detection-error';

  /// Width of the frame sent to the server (height keeps the camera's 2:1).
  static const int sendWidth = 960;
  static const int jpegQuality = 50; // the web version uses 0.5
  static const int minIntervalMs = 100; // minimum gap between frames sent
  static const Duration timeout = Duration(seconds: 3);

  /// Outlines older than this are removed from the screen.
  static const int resultTtlMs = 2000;
  static const double minScore = 0.0;

  /// Douglas-Peucker tolerance in pixels of the sent frame (web uses 1.5).
  static const double simplifyEpsilonPx = 2.0;

  /// true = no network, returns a fake circle in the middle of the frame.
  static const bool useMock = false;
}

// =====================================================================
// Model
// =====================================================================

class Detection {
  const Detection(this.label, this.score, this.points);
  final String label;
  final double score;

  /// Outline, normalized 0..1 across the full equirectangular frame
  /// (x = 0 is the left edge, y = 0 is the top).
  final List<Offset> points;
}

// =====================================================================
// JPEG encoding (background isolate) -> "data:image/jpeg;base64,..."
// Same payload the web version sends in `image`.
// =====================================================================

Future<String> encodeJpegDataUrl(Uint8List rgba, int w, int h) {
  final quality = ApiConfig.jpegQuality;
  return Isolate.run<String>(() {
    final im = img.Image.fromBytes(
      width: w,
      height: h,
      bytes: rgba.buffer,
      bytesOffset: rgba.offsetInBytes,
      numChannels: 4,
    );
    final jpg = img.encodeJpg(im, quality: quality);
    return 'data:image/jpeg;base64,${base64Encode(jpg)}';
  });
}

// =====================================================================
// Socket.IO client
//   emit   'detect-frame'      { frameId, image: dataUrl }
//   listen 'detection-result'  { detections: [{label, confidence, bbox, segments}] }
//   listen 'detection-error'
// One frame is in flight at a time: the next one is sent after the result.
// =====================================================================

class DetectionClient {
  final ValueNotifier<bool> connected = ValueNotifier(false);
  void Function(String message)? onLog;

  io.Socket? _socket;
  Completer<List<Detection>>? _pending;
  int _sentW = 1, _sentH = 1;
  int _frameId = 0;
  bool _closed = false;

  bool get ready => ApiConfig.useMock || connected.value;

  void connect() {
    if (ApiConfig.useMock || _socket != null) return;

    final socket = io.io(
      ApiConfig.serverUrl,
      io.OptionBuilder()
          .setTransports(['websocket'])
          .enableAutoConnect()
          .build(),
    );
    _socket = socket;

    socket.onConnect((_) {
      if (_closed) return;
      connected.value = true;
      onLog?.call('Socket connected: ${ApiConfig.serverUrl}');
    });
    socket.onDisconnect((_) {
      if (_closed) return;
      connected.value = false;
      onLog?.call('Socket disconnected');
      _failPending('socket disconnected');
    });
    socket.onConnectError((e) => onLog?.call('Socket connect error: $e'));

    socket.on(ApiConfig.resultEvent, (data) {
      final p = _pending;
      if (p == null || p.isCompleted) return; // late result, ignore
      _pending = null;
      try {
        p.complete(parseDetections(data, _sentW, _sentH));
      } catch (e) {
        p.completeError(e);
      }
    });

    socket.on(
      ApiConfig.errorEvent,
      (err) => _failPending('server error: $err'),
    );
  }

  void _failPending(String message) {
    final p = _pending;
    _pending = null;
    if (p != null && !p.isCompleted) p.completeError(message);
  }

  Future<List<Detection>> detect(String dataUrl, int w, int h) async {
    if (ApiConfig.useMock) {
      await Future.delayed(const Duration(milliseconds: 80));
      return [
        Detection('mock', 0.99, [
          for (int i = 0; i < 24; i++)
            Offset(
              0.5 + 0.06 * math.cos(i * math.pi / 12),
              0.5 + 0.12 * math.sin(i * math.pi / 12),
            ),
        ]),
      ];
    }

    final socket = _socket;
    if (socket == null || !connected.value) {
      throw 'server not connected';
    }
    _sentW = w;
    _sentH = h;
    final c = Completer<List<Detection>>();
    _pending = c;
    socket.emit(ApiConfig.sendEvent, {'frameId': _frameId++, 'image': dataUrl});
    return c.future.timeout(
      ApiConfig.timeout,
      onTimeout: () {
        if (identical(_pending, c)) _pending = null;
        throw TimeoutException('no response from server');
      },
    );
  }

  void close() {
    _closed = true;
    _socket?.dispose();
    _socket = null;
    connected.dispose();
  }
}

// =====================================================================
// Response parsing
// =====================================================================

List<Detection> parseDetections(dynamic payload, int sentW, int sentH) {
  dynamic root = payload;
  if (root is String) root = jsonDecode(root);
  final items = root is Map ? root['detections'] : root;
  if (items is! List) return const [];

  final out = <Detection>[];
  for (final it in items) {
    if (it is! Map) continue;
    final conf = it['confidence'] ?? it['score'];
    final score = conf is num ? conf.toDouble() : 0.0;
    if (score < ApiConfig.minScore) continue;
    final label = '${it['label'] ?? 'object'}';

    final pts = _outlineOf(it, sentW, sentH);
    if (pts.length < 3) continue;
    out.add(Detection(label, score, pts));
  }
  return out;
}

/// Builds a normalized outline from `segments` (or from `bbox` if there are
/// no segments). Coordinates up to 1.5 are treated as already normalized,
/// larger values as pixels of the frame that was sent.
List<Offset> _outlineOf(Map it, int w, int h) {
  var pts = _toPoints(
    it['segments'] ??
        it['polygon'] ??
        it['points'] ??
        it['outline'] ??
        it['contour'],
  );
  bool fromBox = false;
  if (pts.length < 3) {
    final b = it['bbox'] ?? it['box'];
    if (b is Map &&
        b['x1'] is num &&
        b['y1'] is num &&
        b['x2'] is num &&
        b['y2'] is num) {
      pts = _boxRing(
        (b['x1'] as num).toDouble(),
        (b['y1'] as num).toDouble(),
        (b['x2'] as num).toDouble(),
        (b['y2'] as num).toDouble(),
      );
      fromBox = true;
    } else {
      return const [];
    }
  }

  double maxV = 0;
  for (final p in pts) {
    maxV = math.max(maxV, math.max(p.dx, p.dy));
  }
  final isPixels = maxV > 1.5;

  if (!fromBox) {
    pts = simplifyPolygon(
      pts,
      isPixels ? ApiConfig.simplifyEpsilonPx : ApiConfig.simplifyEpsilonPx / w,
    );
  }
  return isPixels ? [for (final p in pts) Offset(p.dx / w, p.dy / h)] : pts;
}

// A rectangle with extra points along each edge, so it bends correctly when
// it is projected onto the 360 view.
List<Offset> _boxRing(double x1, double y1, double x2, double y2) {
  const n = 10;
  Offset lerp(Offset a, Offset b, int i) =>
      Offset(a.dx + (b.dx - a.dx) * i / n, a.dy + (b.dy - a.dy) * i / n);
  final tl = Offset(x1, y1), tr = Offset(x2, y1);
  final br = Offset(x2, y2), bl = Offset(x1, y2);
  return [
    for (int i = 0; i < n; i++) lerp(tl, tr, i),
    for (int i = 0; i < n; i++) lerp(tr, br, i),
    for (int i = 0; i < n; i++) lerp(br, bl, i),
    for (int i = 0; i < n; i++) lerp(bl, tl, i),
  ];
}

List<Offset> _toPoints(dynamic raw) {
  if (raw is! List || raw.isEmpty) return const [];
  final first = raw.first;
  if (first is num) {
    return [
      for (int i = 0; i + 1 < raw.length; i += 2)
        Offset((raw[i] as num).toDouble(), (raw[i + 1] as num).toDouble()),
    ];
  }
  if (first is Map) {
    return [
      for (final p in raw)
        if (p is Map && p['x'] is num && p['y'] is num)
          Offset((p['x'] as num).toDouble(), (p['y'] as num).toDouble()),
    ];
  }
  if (first is List) {
    if (first.length == 2 && first[0] is num) {
      return [
        for (final p in raw)
          if (p is List && p.length >= 2 && p[0] is num && p[1] is num)
            Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()),
      ];
    }
    return _toPoints(first); // nested lists: use the first polygon
  }
  return const [];
}

// Douglas-Peucker. Handles a zero-length base line (closed outlines whose
// first and last point are equal), which would otherwise divide by zero.
List<Offset> simplifyPolygon(List<Offset> pts, double eps) {
  if (pts.length < 3) return pts;
  final a = pts.first, b = pts.last;
  double dMax = 0;
  int idx = 0;
  for (int i = 1; i < pts.length - 1; i++) {
    final d = _distToLine(pts[i], a, b);
    if (d > dMax) {
      dMax = d;
      idx = i;
    }
  }
  if (dMax > eps) {
    final l = simplifyPolygon(pts.sublist(0, idx + 1), eps);
    final r = simplifyPolygon(pts.sublist(idx), eps);
    return [...l.sublist(0, l.length - 1), ...r];
  }
  return [a, b];
}

double _distToLine(Offset p, Offset a, Offset b) {
  final dx = b.dx - a.dx, dy = b.dy - a.dy;
  final len2 = dx * dx + dy * dy;
  if (len2 == 0) return (p - a).distance;
  final t = ((p.dx - a.dx) * dx + (p.dy - a.dy) * dy) / len2;
  final px = a.dx + t * dx, py = a.dy + t * dy;
  final ex = p.dx - px, ey = p.dy - py;
  return math.sqrt(ex * ex + ey * ey);
}

// =====================================================================
// Overlay: draws outlines on top of the 360 view.
//
// Points live in equirectangular space, so they are projected with the same
// yaw / pitch / fov as the shader. Outlines stay glued to the objects while
// you pan and zoom.
// =====================================================================

class OutlinePainter extends CustomPainter {
  OutlinePainter(this.dets, this.yaw, this.pitch, this.fov, Listenable repaint)
    : super(repaint: repaint);

  final ValueNotifier<List<Detection>> dets;
  final double Function() yaw;
  final double Function() pitch;
  final double Function() fov;

  final Paint _stroke = Paint()
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.5
    ..strokeJoin = StrokeJoin.round;
  final Paint _fill = Paint()..style = PaintingStyle.fill;
  final Paint _labelBg = Paint();

  // Same palette as the web version, in order.
  static const _palette = [
    Color(0xFFFF4D4F),
    Color(0xFF4ADE80),
    Color(0xFF38BDF8),
    Color(0xFFF59E0B),
    Color(0xFFC084FC),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final list = dets.value;
    if (list.isEmpty) return;

    final y = yaw(), p = pitch();
    final f = 1.0 / math.tan(fov() * 0.5);
    final cy = math.cos(y), sy = math.sin(y);
    final cp = math.cos(p), sp = math.sin(p);
    final aspect = size.width / size.height;

    // Inverse of the shader's mapping: equirect (u,v) -> screen pixel.
    Offset? project(Offset uv) {
      final lon = (uv.dx - 0.5) * 2 * math.pi;
      final lat = (0.5 - uv.dy) * math.pi;
      final cl = math.cos(lat);
      final x = cl * math.sin(lon);
      final yv = math.sin(lat);
      final z = cl * math.cos(lon);
      // undo yaw, then undo pitch
      final x1 = x * cy - z * sy;
      final z1 = x * sy + z * cy;
      final y2 = yv * cp + z1 * sp;
      final z2 = -yv * sp + z1 * cp;
      if (z2 <= 0.02) return null; // behind the camera
      final px = f * x1 / z2;
      final py = -f * y2 / z2;
      return Offset(
        (px / aspect + 1) * 0.5 * size.width,
        (py + 1) * 0.5 * size.height,
      );
    }

    for (int i = 0; i < list.length; i++) {
      final d = list[i];
      final path = Path();
      bool pen = false;
      bool allVisible = true;
      Offset? labelAt;
      for (final uv in d.points) {
        final s = project(uv);
        if (s == null) {
          pen = false;
          allVisible = false;
          continue;
        }
        if (pen) {
          path.lineTo(s.dx, s.dy);
        } else {
          path.moveTo(s.dx, s.dy);
        }
        pen = true;
        if (labelAt == null || s.dy < labelAt.dy) labelAt = s;
      }
      if (labelAt == null) continue; // fully off screen / behind

      final color = _palette[i % _palette.length];
      if (allVisible) {
        path.close();
        _fill.color = color.withAlpha(46);
        canvas.drawPath(path, _fill);
      }
      _stroke.color = color;
      canvas.drawPath(path, _stroke);

      final tp = TextPainter(
        text: TextSpan(
          text: '${d.label} ${(d.score * 100).round()}%',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final maxX = math.max(0.0, size.width - tp.width - 8);
      final maxY = math.max(0.0, size.height - tp.height - 2);
      final pos = Offset(
        labelAt.dx.clamp(6.0, maxX + 6).toDouble(),
        (labelAt.dy - tp.height - 4).clamp(0.0, maxY).toDouble(),
      );
      _labelBg.color = const Color(0xB3000000); // rgba(0,0,0,0.7) like the web
      canvas.drawRect(
        Rect.fromLTWH(pos.dx - 6, pos.dy, tp.width + 12, tp.height + 2),
        _labelBg,
      );
      tp.paint(canvas, Offset(pos.dx, pos.dy + 1));
    }
  }

  @override
  bool shouldRepaint(covariant OutlinePainter old) => false; // driven by `repaint`
}
