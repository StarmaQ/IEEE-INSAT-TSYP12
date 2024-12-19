import 'package:adhd_diagnosis/controllers/data_controller.dart';
import 'package:flutter/material.dart';
import 'package:flame/game.dart';
import 'package:adhd_diagnosis/common_libs.dart';
import 'package:adhd_diagnosis/controllers/game_loader_controller.dart';

// games
import 'package:adhd_diagnosis/ui/games/flappy_bird.dart';

final dataController = DataController(baseUrl: "http://10.0.2.2:5000");

final Map<String, Map<String, dynamic>> games = {
  'Flappy Bird': {
    'image': 'https://via.placeholder.com/150',
    'screen': GameWidget(game: FlappyBirdGame(dataController: dataController)),
  },
};

class GameMenuScreen extends StatelessWidget {
  const GameMenuScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Game Menu'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: GridView.builder(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            crossAxisSpacing: 8.0,
            mainAxisSpacing: 8.0,
            childAspectRatio: 1,
          ),
          itemCount: 1,
          itemBuilder: (context, index) {
            final String gameName = games.keys.elementAt(index);

            final Map<String, dynamic> game = games[gameName]!;

            return GameTile(
              game: game,
            );
          },
        ),
      ),
    );
  }
}

class GameTile extends StatelessWidget {
  final Map<String, dynamic> game;

  const GameTile({
    super.key,
    required this.game,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) {
              return GameLoaderController.loadGame(
                game: game["screen"],
                onGazeDetected: (data) {
                  // Handle gaze data detection
                  print("Gaze detected: $data");
                },
                onError: (error) {
                  // Handle error
                  print("Error: $error");
                },
              );
            },
          ),
        );
      },
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8.0),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 10,
              offset: Offset(5, 5), // Shadow position
            ),
            BoxShadow(
              color: Colors.white.withOpacity(0.7),
              blurRadius: 10,
              offset: Offset(-5, -5), // Highlight
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(top: Radius.circular(8.0)),
                child: Image.asset(
                  "assets/flappybird/icon.png",
                  fit: BoxFit.cover,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                "Flappy Bird",
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
