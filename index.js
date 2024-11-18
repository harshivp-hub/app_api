const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const { spawnSync } = require('child_process');
const path = require('path');
process.env.TF_CPP_MIN_LOG_LEVEL = '2';  // Suppresses most TensorFlow logs
process.env.TF_ENABLE_ONEDNN_OPTS = '0'; // Disables oneDNN optimizations
require('dotenv').config();

// Setup express app
const app = express();
app.use(express.json());
app.use(cors());

// MongoDB Connection
const MONGO_URI = process.env.MONGO_URI;
mongoose.connect(MONGO_URI);

const JWT_SECRET = process.env.JWT_SECRET;


// Spotify credentials
const SPOTIFY_CLIENT_ID = process.env.SPOTIFY_CLIENT_ID;
const SPOTIFY_CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET;


// Affirmation Schema
const affirmationSchema = new mongoose.Schema({
  Affirmation: String,
});
const Affirmation = mongoose.model('affirmation', affirmationSchema);

// Mood Schema
const moodSchema = new mongoose.Schema({
  userId: String,
  mood: String,
  date: { type: Date, default: Date.now },
});
const Mood = mongoose.model('Mood', moodSchema);

// User Schema
const userSchema = new mongoose.Schema({
  userId: { type: String, unique: true },
  name: String,
  email: { type: String, unique: true },
  password: String,
  age: Number,
  gender: String,
  mobileNumber: String,
});
const User = mongoose.model('User', userSchema);

// Task Schema
const taskSchema = new mongoose.Schema({
  userId: String,
  task: String,
  date: String,
  isCompleted: Boolean,
});
const Task = mongoose.model('Task', taskSchema);

// Fetch a random affirmation API
app.get('/api/affirmation/random', async (req, res) => {
  try {
    const count = await Affirmation.countDocuments();
    if (count === 0) {
      return res.status(404).json({ error: 'No affirmations found' });
    }

    const randomIndex = Math.floor(Math.random() * count);
    const randomAffirmation = await Affirmation.findOne().skip(randomIndex);

    res.json(randomAffirmation);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching random affirmation' });
  }
});

// Register API
app.post('/api/register', async (req, res) => {
  const { name, email, password, age, gender, mobileNumber } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const userId = uuidv4();

    const user = new User({
      userId,
      name,
      email,
      password: hashedPassword,
      age,
      gender,
      mobileNumber,
    });

    await user.save();
    res.status(201).json({ message: 'User registered successfully', userId });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: 'Error registering user' });
  }
});

// Login API
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: 'User not found' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: user.userId }, JWT_SECRET, { expiresIn: '1h' });
    res.json({ token, message: 'Login successful', userId: user.userId });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Get User Details by userId
app.get('/api/user/:userId', async (req, res) => {
  const { userId } = req.params;
  try {
    const user = await User.findOne({ userId });
    if (user) {
      res.json({ name: user.name });
    } else {
      res.status(404).json({ error: 'User not found' });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Add Task API
app.post('/api/tasks', async (req, res) => {
  const { userId, task, date } = req.body;

  try {
    const newTask = new Task({
      userId,
      task,
      date,
      isCompleted: false,
    });

    await newTask.save();
    res.status(201).json({ message: 'Task added successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error adding task' });
  }
});

// Fetch Tasks API
app.get('/api/tasks/:userId', async (req, res) => {
  const { userId } = req.params;

  try {
    const tasks = await Task.find({ userId });
    res.json(tasks);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching tasks' });
  }
});

// Toggle task completion and delete if completed
app.put('/api/tasks/:taskId', async (req, res) => {
  const { taskId } = req.params;
  const { isCompleted } = req.body;

  try {
    if (isCompleted) {
      await Task.findByIdAndDelete(taskId);
      res.status(200).json({ message: 'Task completed and deleted' });
    } else {
      await Task.findByIdAndUpdate(taskId, { isCompleted });
      res.status(200).json({ message: 'Task completion status updated' });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error updating task' });
  }
});

// Function to get Spotify token
async function getSpotifyAccessToken() {
  try {
    const authResponse = await axios.post(
      'https://accounts.spotify.com/api/token',
      new URLSearchParams({
        grant_type: 'client_credentials',
      }),
      {
        headers: {
          'Authorization': `Basic ${Buffer.from(`${SPOTIFY_CLIENT_ID}:${SPOTIFY_CLIENT_SECRET}`).toString('base64')}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      }
    );
    return authResponse.data.access_token;
  } catch (error) {
    console.error('Error getting Spotify access token', error);
    throw new Error('Failed to get Spotify access token');
  }
}

// Spotify playlist IDs
const playlistIds = {
  'yoga_and_meditation': '37i9dQZF1DWZqd5JICZI0u',
  'stress_relief': '37i9dQZF1DWXe9gFZP0gtP',
  'nature_and_noise': '37i9dQZF1DXdzGIPNRTvyN',
  'sleep': '37i9dQZF1DWZd79rJ6a7lp',
  'flute_meditation': '37i9dQZF1DXcj8Mdu8qUVH',
  'workout_playlist': '2SM6rniZl84fEyMCB5KMQB',
  'emotional_wellbeing': '5EKxYpwDPZHIH4psFz8B6s',
};

// Function to fetch tracks from a playlist
async function getPlaylistTracks(accessToken, playlistId) {
  try {
    const response = await axios.get(`https://api.spotify.com/v1/playlists/${playlistId}/tracks`, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      params: {
        limit: 10, // Fetch 10 songs
      },
    });
    return response.data.items.map((item) => item.track); // Extract track details
  } catch (error) {
    console.error(`Error fetching playlist ${playlistId} tracks`, error);
    throw new Error('Failed to fetch playlist tracks');
  }
}

// Fetch recommendations from specific playlists
app.get('/api/recommendations', async (req, res) => {
  try {
    const accessToken = await getSpotifyAccessToken();

    const playlistTracksPromises = Object.values(playlistIds).map((id) =>
      getPlaylistTracks(accessToken, id)
    );

    const playlistTracksResults = await Promise.all(playlistTracksPromises);

    const wellnessRecommendations = {};
    Object.keys(playlistIds).forEach((key, index) => {
      wellnessRecommendations[key] = playlistTracksResults[index];
    });

    res.json(wellnessRecommendations);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching recommendations from Spotify' });
  }
});

// Endpoint to store mood data
app.post('/api/mood', async (req, res) => {
  const { userId, mood } = req.body;

  try {
    const newMood = new Mood({
      userId, // Use userId from request body
      mood,
    });

    await newMood.save();
    res.status(201).json({ message: 'Mood saved successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error saving mood' });
  }
});

// Route to get mood logs for a specific user
app.get('/api/mood/:userId', async (req, res) => {
  const { userId } = req.params;
  try {
    const moods = await Mood.find({ userId }).sort({ date: -1 });
    res.json(moods);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching mood logs' });
  }
});

// Journal Entry Schema
const journalSchema = new mongoose.Schema({
  userId: String,
  journalText: String,
  date: { type: Date, default: Date.now },
  emotions: [String],
  topics: [String],
});
const JournalEntry = mongoose.model('JournalEntry', journalSchema);

// Other schemas (e.g., Mood, User, Task, Affirmation)
// Assume they are the same as the original code

// Analyze Journal Entry API
// Analyze Journal Entry API
// Analyze Journal Entry API
// Analyze Journal Entry API
app.post('/api/analyze_journal', async (req, res) => {
  const { userId, journalText } = req.body;

  if (!userId || !journalText) {
    return res.status(400).json({ message: 'User ID and journal text are required.' });
  }

  try {
    const pythonScriptPath = path.join(__dirname, 'analyze_journal.py');
    const pythonProcess = spawnSync('python', [pythonScriptPath, JSON.stringify(journalText)]);

    const output = pythonProcess.stdout.toString().trim();
    const error = pythonProcess.stderr.toString();

    if (error) {
      console.error(`Python Error: ${error}`);
      return res.status(500).json({ message: 'Error processing journal entry.', error });
    }

    // Check if output is valid JSON and ignore any non-JSON output
    let result;
    try {
      const jsonOutput = output.startsWith('{') ? output : output.substring(output.indexOf('{'));
      result = JSON.parse(jsonOutput);
    } catch (parseError) {
      console.error(`Error parsing Python output: ${parseError.message}, output: ${output}`);
      return res.status(500).json({ message: 'Failed to parse Python output.', error: parseError.message });
    }

    const journalEntry = new JournalEntry({
      userId,
      journalText,
      emotions: result.emotions,
      topics: result.topics,
    });

    await journalEntry.save();

    res.status(201).json({
      message: 'Journal entry analyzed and saved successfully.',
      data: journalEntry,
    });
  } catch (err) {
    console.error(`Internal server error: ${err.message}`);
    res.status(500).json({ message: 'Internal server error.' });
  }
});

app.get('/api/journal/:userId', async (req, res) => {
  const { userId } = req.params;
  
  try {
    const journalEntries = await JournalEntry.find({ userId }).sort({ date: -1 });
    res.json(journalEntries);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching journal entries' });
  }
});

// Body Data Schema
const bodyDataSchema = new mongoose.Schema({
  userId: { type: String, unique: true },
  height: Number,
  weight: Number,
});
const BodyData = mongoose.model('BodyData', bodyDataSchema);

// Get body data (height and weight) by userId
app.get('/api/user/:userId/body-data', async (req, res) => {
  const { userId } = req.params;

  try {
    const bodyData = await BodyData.findOne({ userId });

    if (!bodyData) {
      return res.status(404).json({ error: 'No body data found for this user' });
    }

    res.json(bodyData);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error fetching body data' });
  }
});

// Save or update body data for a user
app.post('/api/user/:userId/body-data', async (req, res) => {
  const { userId } = req.params;
  const { height, weight } = req.body;

  try {
    let bodyData = await BodyData.findOne({ userId });

    if (bodyData) {
      // Update existing data
      bodyData.height = height;
      bodyData.weight = weight;
      await bodyData.save();
    } else {
      // Create new data
      bodyData = new BodyData({ userId, height, weight });
      await bodyData.save();
    }

    res.status(200).json({ message: 'Body data saved successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error saving body data' });
  }
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
