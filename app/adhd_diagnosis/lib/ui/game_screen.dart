import 'package:adhd_diagnosis/ui/games/flappy_bird.dart';
import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import '../controllers/data_controller.dart';
import 'dart:convert';

class GameScreen extends StatefulWidget {
  final GameWidget<FlappyBirdGame> game;

  const GameScreen({
    super.key,
    required this.game,
  });

  @override
  _GameScreenState createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  Size _imageSize = Size(1, 1);

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      child: Stack(
        children: [
          Positioned.fill(
            child: widget.game
          ),

          Stack(
            children: [
              // Top bar image
              Positioned(
                top: 0,
                left: 0,
                right: 0,
                child: Image.asset(
                  'assets/game_ui/topbar.png', // Replace with your image asset path
                  fit: BoxFit.cover,
                  height: 100, // Adjust the height of the top bar
                ),
              ),

              // Center the score box widget inside the top bar
              Positioned(
                top: 10, // Adjust this value to center the widget vertically within the top bar
                left: 0,
                right: 0,
                child: Center(
                  child: Stack(
                    children: [
                      // Score box image
                      Image.asset(
                        'assets/game_ui/score.png',
                        width: 100,
                        height: 50,
                      ),

                      Positioned(
                        top: 12,
                        left: 60,
                        child: ValueListenableBuilder<int>(
                          valueListenable: widget.game.game!.score,
                          builder: (context, value, child) {
                            return Text(
                              '$value',
                              style: TextStyle(
                                fontSize: 24,
                                color: Colors.white,
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              Positioned(
                right: 200,
                top: 8,
                child: Image.asset(
                  'assets/game_ui/points.png',
                  width: 60,
                  height: 50,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    super.dispose();
  }
}
