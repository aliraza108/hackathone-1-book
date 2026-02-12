# Physical AI & Humanoid Robotics Textbook

Welcome to the Physical AI & Humanoid Robotics textbook - a complete educational resource with integrated Qwen-powered chatbot, retro-themed UI, and comprehensive coverage of modern robotics concepts.

## 📚 About

This textbook covers 13 weeks of material on Physical AI and Humanoid Robotics, designed for advanced Computer Science and Engineering students. The content is derived from the official "Physical AI & Humanoid Robotics" curriculum and enhanced with interactive elements and AI-powered assistance.

## ✨ Features

- **13-week curriculum** covering ROS 2, Gazebo, NVIDIA Isaac, and VLA systems
- **Integrated Qwen LLM-powered chatbot** for educational assistance
- **Retro 80s computing aesthetic** with modern UX
- **Interactive code examples** and simulations
- **Hardware requirements** and project guidelines
- **Capstone project**: Autonomous humanoid with voice commands
- **Text selection queries**: Click any text to ask the chatbot for explanations

## 🛠️ Tech Stack

- **Frontend**: Docusaurus v3.x with custom retro theme
- **AI/Chatbot**: Qwen LLM with RAG pipeline
- **Backend**: FastAPI (Python)
- **Database**: Neon Serverless Postgres, Qdrant Vector Store
- **Deployment**: GitHub Pages (frontend), Railway/Render (backend)

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.8+
- Access to Qwen API (for chatbot functionality)

### Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm start
```

### Backend Setup
```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Start the backend server
uvicorn app.main:app --reload
```

### Environment Variables
Create a `.env` file in the backend directory with:
```env
# Qwen API
QWEN_API_KEY=your_qwen_api_key
QWEN_MODEL=qwen-turbo
QWEN_API_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=physical-ai-textbook

# Database
DATABASE_URL=your_database_url
```

## 📖 Content Structure

The textbook is organized into four modules:

1. **Module 1**: The Robotic Nervous System (ROS 2) - Weeks 1-5
2. **Module 2**: The Digital Twin (Gazebo & Unity) - Weeks 6-7
3. **Module 3**: The AI-Robot Brain (NVIDIA Isaac) - Weeks 8-10
4. **Module 4**: Vision-Language-Action (VLA) - Weeks 11-13

Each week includes theoretical concepts, practical examples, and hands-on exercises.

## 🤖 Chatbot Integration

The integrated Qwen-powered chatbot can:
- Answer questions about textbook content using RAG
- Explain complex concepts with examples
- Provide code explanations
- Cite specific chapters and sections
- Support text selection queries ("Explain this section")

## 🎨 Retro Theme

The UI features an 80s computing aesthetic with:
- CRT scanline effects
- Terminal-style interfaces
- Neon green color scheme
- Glitch animations
- ASCII art elements

## 📁 Project Structure

```
physical-ai-textbook/
├── docs/                      # Docusaurus content
│   ├── intro.md
│   ├── module-1/             # ROS 2 fundamentals
│   ├── module-2/             # Simulation (Gazebo/Unity)
│   ├── module-3/             # AI (Isaac SDK)
│   ├── module-4/             # VLA & Humanoid Robotics
│   └── appendix/             # Hardware, Glossary, Resources
├── src/                       # Frontend components
│   ├── components/           # React components
│   ├── css/                  # Styling
│   ├── hooks/                # React hooks
│   └── utils/                # Utility functions
├── backend/                   # FastAPI backend
│   ├── app/                  # Application code
│   ├── services/             # Business logic
│   ├── models/               # Data models
│   └── scripts/              # Utility scripts
├── static/                    # Static assets
└── content.pdf               # Source textbook content
```

## 🧪 Testing

To run backend tests:
```bash
cd backend
python -m pytest
```

## 🚀 Deployment

### Frontend (GitHub Pages)
The frontend automatically deploys to GitHub Pages when pushed to the main branch via GitHub Actions.

### Backend
Deploy the backend to Railway, Render, or Fly.io with the required environment variables.

## 🤝 Contributing

We welcome contributions to improve the textbook content, fix bugs, or enhance features. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Special thanks to the Physical AI & Robotics community for inspiration
- Qwen team for the excellent LLM capabilities
- Docusaurus team for the fantastic documentation framework
- All contributors who helped make this project possible