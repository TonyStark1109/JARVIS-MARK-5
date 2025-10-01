const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'perplexica',
    timestamp: new Date().toISOString()
  });
});

// Search endpoint
app.post('/search', async (req, res) => {
  try {
    const { query, type = 'web' } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Mock search results for now
    const results = {
      query,
      type,
      results: [
        {
          title: `Search results for: ${query}`,
          url: 'https://example.com',
          snippet: `This is a mock search result for "${query}". Perplexica is integrated with JARVIS.`,
          score: 0.95
        }
      ],
      timestamp: new Date().toISOString()
    };

    res.json(results);
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// AI-powered search endpoint
app.post('/ai-search', async (req, res) => {
  try {
    const { query, context } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Mock AI search results
    const aiResults = {
      query,
      context,
      aiResponse: `Based on your query "${query}", here's what I found: This is an AI-powered search response from Perplexica integrated with JARVIS.`,
      sources: [
        {
          title: 'JARVIS Documentation',
          url: 'https://jarvis.example.com',
          relevance: 0.9
        }
      ],
      timestamp: new Date().toISOString()
    };

    res.json(aiResults);
  } catch (error) {
    console.error('AI search error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Perplexica server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🔍 Search endpoint: http://localhost:${PORT}/search`);
  console.log(`🤖 AI Search endpoint: http://localhost:${PORT}/ai-search`);
});

module.exports = app;
