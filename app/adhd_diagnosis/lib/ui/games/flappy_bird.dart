import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:flame/cache.dart';
import 'package:flame/collisions.dart';
import 'package:flame/components.dart';
import 'package:flame/events.dart';
import 'package:flame/flame.dart';
import 'package:flame/game.dart';
import 'package:flame/image_composition.dart';
import 'package:flame/parallax.dart';
import 'package:flame/sprite.dart';
import 'package:flame_audio/flame_audio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/painting.dart';
import 'package:adhd_diagnosis/controllers/game.dart';
import 'package:adhd_diagnosis/controllers/data_controller.dart';

class BonusZone extends PositionComponent with CollisionCallbacks {
  BonusZone({super.size});

  @override
  FutureOr<void> onLoad() {
    add(RectangleHitbox(size: size));
    return super.onLoad();
  }

  @override
  void onCollisionEnd(PositionComponent other) {
    super.onCollisionEnd(other);
    if (other is Player) {
      other.score++;
      removeFromParent();
      FlameAudio.play("point.wav");
      print("Bonus: ${other.score}");
    }
  }
}

class Player extends SpriteAnimationComponent with CollisionCallbacks {
  var isDead = false;
  var score = 0;

  Player({super.animation});

  @override
  FutureOr<void> onLoad() {
    add(RectangleHitbox(size: size));
    return super.onLoad();
  }

  @override
  void onCollisionStart(
      Set<Vector2> intersectionPoints, PositionComponent other) {
    super.onCollisionStart(intersectionPoints, other);
    if (other is PipeComponent) {
      FlameAudio.play("hit.wav");
      isDead = true;
    }
  }
}

class FlyingObstacle extends SpriteComponent {
  final double speed;

  FlyingObstacle({required this.speed, required Sprite sprite, required Vector2 size})
      : super(sprite: sprite, size: size);

  @override
  void update(double dt) {
    super.update(dt);
    position.x -= speed * dt;

    // Remove the obstacle if it goes out of bounds
    if (position.x + size.x < 0) {
      removeFromParent();
    }
  }
}

class PipeComponent extends PositionComponent with CollisionCallbacks {
  final bool isUpsideDown;
  final Images? images;
  PipeComponent({this.isUpsideDown = false, this.images, super.size});
  @override
  FutureOr<void> onLoad() async {
    final nineBox = NineTileBox(
        await Sprite.load("pipe-green.png", images: images))
      ..setGrid(leftWidth: 10, rightWidth: 10, topHeight: 60, bottomHeight: 60);
    final spriteCom = NineTileBoxComponent(nineTileBox: nineBox, size: size);
    if (isUpsideDown) {
      spriteCom.flipVerticallyAroundCenter();
    }
    spriteCom.anchor = Anchor.topLeft;

    add(spriteCom);

    add(RectangleHitbox(size: size));
    return super.onLoad();
  }
}

class FlappyBirdGame extends ADHDGame with TapDetector, HasCollisionDetection {
  final images = Images(prefix: "assets/flappybird/sprites/");
  var gameSpeed = 60.0;
  final pipeFullSize = Vector2(52.0, 460.0);
  late PositionComponent _pipeLayer;
  final List<FlyingObstacle> _obstacles = [];

  FlappyBirdGame({required DataController dataController})
      : super(dataController: dataController);

  late Timer _obstacleTimer;
  late Timer _sendBirdPosition;

  @override
  FutureOr<void> onLoad() async {
    FlameAudio.updatePrefix("assets/flappybird/audios/");
    // FlameAudio.bgm.play("wing.wav");

    await setupBg();
    await setupBird();
    await setupScoreLabel();

    setupObstacleSpawner();
    _startTapTimer();
    _startBirdTimer();

    resetGame();
    return super.onLoad();
  }

  
  @override
  void update(double dt) {
    super.update(dt);
    updateBird(dt);
    updatePipes(dt);
    updateScoreLabel();

    _obstacleTimer.update(dt); // Update the obstacle timer
    _tapTimer.update(dt);
    _sendBirdPosition.update(dt);

    if (_birdComponent.isDead) {
      FlameAudio.play("die.wav");
      gameOver();
    }
  }

  int _tapCount = 0; // To count the number of taps in a second
  late Timer _tapTimer; // Timer to send the data every second

  @override
  void onTap() {
    super.onTap();
    FlameAudio.play("swoosh.wav");
    _birdYVelocity = -120;

    _tapCount++; // Increment the tap count every time the screen is tapped
   
    
  }

  void _startBirdTimer() {
    _sendBirdPosition = Timer(0.1, onTick: () {
      final DateTime timestamp = DateTime.now();

      dataController.sendData('submit-data', {
        'data_type': 'bird_position',
        'value': _birdComponent.position.x.toString() + "," + _birdComponent.position.y.toString(),
        'timestamp': timestamp.toIso8601String(),
      }).then((response) {
        print('Score sent successfully: $response');
      }).catchError((error) {
        print('Error sending score data: $error');
      });

      _sendBirdPosition.start();
    })..start();

  }



  void _startTapTimer() {

    _tapTimer = Timer(1, onTick: () {
      final DateTime timestamp = DateTime.now();
      print("SENDING DATAAA");
      dataController.sendData('submit-data', {
        'data_type': 'player_input',
        'value': _tapCount,
        'timestamp': timestamp.toIso8601String(),
      }).then((response) {
        print('Score sent successfully: $response');
      }).catchError((error) {
        print('Error sending score data: $error');
      });
      dataController.sendData('submit-data', {
        'data_type': 'score',
        'value': score.value,
        'timestamp': timestamp.toIso8601String(),
      }).then((response) {
        print('Score sent successfully: $response');
      }).catchError((error) {
        print('Error sending score data: $error');
      });

      _tapTimer.start();

      _tapCount = 0;
    })..start();

  }

  @override
  void onDispose() {
    super.onDispose();
    FlameAudio.bgm.stop();
  }



  setupObstacleSpawner() {
    _obstacleTimer = Timer(
      Random().nextDouble() * 6 + 2, // Random interval between 2 to 5 seconds
      onTick: () {
        final DateTime timestamp = DateTime.now();

        dataController.sendData('submit-data', {
          'data_type': 'obstacle_spawn',
          'value': 'spawned',
          'timestamp': timestamp.toIso8601String(),
        }).then((response) {
          print('Score sent successfully: $response');
        }).catchError((error) {
          print('Error sending score data: $error');
        });

        spawnObstacle();
        _obstacleTimer.start(); // Restart the timer
      },
    )..start(); // Start the timer
  }

  void spawnObstacle() async {
    // Load a sprite for the obstacle
    final obstacleSprite = await Sprite.load('obstacle.png', images: images);

    // Randomize the vertical position of the obstacle
    final obstacleY = Random().nextDouble() * (size.y - 112); // Avoid the bottom platform

    // Create the obstacle
    final obstacle = FlyingObstacle(
      speed: gameSpeed * 1.5, // Adjust speed as needed
      sprite: obstacleSprite,
      size: Vector2(50, 30), // Adjust size as needed
    )
      ..position = Vector2(size.x, obstacleY)
      ..anchor = Anchor.center;

    add(obstacle);
    _obstacles.add(obstacle);
  }

  setupBg() async {
    final bgComponent = await loadParallaxComponent(
        [ParallaxImageData("background-day.png")],
        baseVelocity: Vector2(5, 0), images: images);
    add(bgComponent);

    _pipeLayer = PositionComponent();
    add(_pipeLayer);

    final bottomBgComponent = await loadParallaxComponent(
        [ParallaxImageData("base.png")],
        baseVelocity: Vector2(gameSpeed, 0),
        images: images,
        alignment: Alignment.bottomLeft,
        repeat: ImageRepeat.repeatX,
        fill: LayerFill.none);
    add(bottomBgComponent);
  }

  // bird
  var _birdYVelocity = 0.0;
  final _gravity = 250.0;
  late Player _birdComponent;
  setupBird() async {
    List<Sprite> redBirdSprites = [
      await Sprite.load("yellowbird-downflap.png", images: images),
      await Sprite.load("yellowbird-midflap.png", images: images),
      await Sprite.load("yellowbird-upflap.png", images: images)
    ];
    final anim = SpriteAnimation.spriteList(redBirdSprites, stepTime: 0.2);
    _birdComponent = Player(animation: anim);
    add(_birdComponent);
  }

  updateBird(double dt) {
    _birdYVelocity += dt * _gravity;

    final birdNewY = _birdComponent.position.y + _birdYVelocity * dt;
    _birdComponent.position = Vector2(_birdComponent.position.x, birdNewY);
    _birdComponent.anchor = Anchor.center;
    final angle = clampDouble(_birdYVelocity / 180, -pi * 0.25, pi * 0.25);
    _birdComponent.angle = angle;

    if (birdNewY > size.y) {
      gameOver();
    }
  }

  // scoreLabel
  List<ui.Image> _numSprites = [];
  setupScoreLabel() async {

  
  }

  updateScoreLabel() async {
    score.value = _birdComponent.score;
  }

  // pipe
  final _pipes = [];
  final _bonusZones = [];
  createPipe() {
    const pipeSpace = 220.0; // the space of two pipe group
    const minPipeHeight = 120.0; // pipe min height
    const gapHeight = 160.0; // the gap length of two pipe 
    const baseHeight = 112.0; // the bottom platform height
    const gapMaxRandomRange = 300; // gap position max random range
    var lastPipePos = _pipes.lastOrNull?.position.x ?? size.x - pipeSpace;
    lastPipePos += pipeSpace;

    final gapCenterPos = min(gapMaxRandomRange,
                size.y - minPipeHeight * 2 - baseHeight - gapHeight) *
            Random().nextDouble() +
        minPipeHeight +
        gapHeight * 0.5;

    PipeComponent topPipe =
        PipeComponent(images: images, isUpsideDown: true, size: pipeFullSize)
          ..position = Vector2(
              lastPipePos, (gapCenterPos - gapHeight * 0.5) - pipeFullSize.y);
    _pipeLayer.add(topPipe);
    _pipes.add(topPipe);

    PipeComponent bottomPipe =
        PipeComponent(images: images, isUpsideDown: false, size: pipeFullSize)
          ..size = pipeFullSize
          ..position = Vector2(lastPipePos, gapCenterPos + gapHeight * 0.5);
    _pipeLayer.add(bottomPipe);
    _pipes.add(bottomPipe);

    final bonusZone = BonusZone(size: Vector2(pipeFullSize.x, gapHeight))
      ..position = Vector2(lastPipePos, gapCenterPos - gapHeight * 0.5);
    add(bonusZone);
    _bonusZones.add(bonusZone);
  }

  updatePipes(double dt) {
    for (final pipe in _pipes) {
      pipe.position =
          Vector2(pipe.position.x - dt * gameSpeed, pipe.position.y);
    }
    for (final bonusZone in _bonusZones) {
      bonusZone.position =
          Vector2(bonusZone.position.x - dt * gameSpeed, bonusZone.position.y);
    }
    _pipes.removeWhere((element) {
      final remove = element.position.x < -100;
      if (remove) {
        element.removeFromParent();
      }
      return remove;
    });
    _bonusZones.removeWhere((element) {
      final remove = element.position.x < -100;
      if (remove) {
        element.removeFromParent();
      }
      return remove;
    });

    if ((_pipes.lastOrNull?.position.x ?? 0) < size.x) {
      createPipe();
    }
  }

  gameOver() {
    FlameAudio.play("die.wav");
    resetGame();
  }

  resetGame() {
    _birdComponent.isDead = false;
    _birdComponent.score = 0;
    _birdComponent.position = Vector2(size.x * 0.3, size.y * 0.5);
    _birdYVelocity = 0.0;
    for (var element in _pipes) {
      element.removeFromParent();
    }
    _pipes.clear();

    for (var element in _bonusZones) {
      element.removeFromParent();
    }


    for (var element in _obstacles) {
      element.removeFromParent();
    }

    _bonusZones.clear();
  }

}
