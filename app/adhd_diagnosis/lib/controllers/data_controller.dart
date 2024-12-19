import 'dart:convert';
import 'package:http/http.dart' as http;

class DataController {
  final String baseUrl; // The base URL of your Flask backend

  DataController({required this.baseUrl});

  /// Sends data to the Flask backend via a POST request
  Future<Map<String, dynamic>> sendData(String endpoint, Map<String, dynamic> data) async {
    try {
      final url = Uri.parse('$baseUrl/$endpoint');

      // Send the POST request with JSON-encoded data
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(data),
      );

      // Check if the request was successful
      if (response.statusCode == 200 || response.statusCode == 201) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to send data: ${response.statusCode} ${response.body}');
      }
    } catch (e) {
      rethrow;
    }
  }
}

/// Example usage:
void main() async {
  final controller = DataController(baseUrl: 'http://your-flask-backend.com');

  try {
    final response = await controller.sendData('submit-data', {
      'name': 'John Doe',
      'email': 'john.doe@example.com',
      'age': 25,
    });

    print('Response from backend: $response');
  } catch (e) {
    print('Error: $e');
  }
}
