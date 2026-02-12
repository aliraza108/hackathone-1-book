# Physical AI & Humanoid Robotics Textbook - Specification

## Project Overview
Create a complete Docusaurus-based educational textbook for "Physical AI & Humanoid Robotics" with an integrated RAG chatbot using Qwen LLM, retro-themed UI, and full deployment pipeline using Spec-Kit Plus workflow.

## Core Deliverables
1. **Docusaurus Educational Textbook** - 13-week Physical AI & Humanoid Robotics course
2. **Integrated RAG Chatbot** - Qwen LLM + OpenAI ChatKit SDK + FastAPI + Neon Postgres + Qdrant
3. **Retro-Themed UI** - 80s computing aesthetics with modern UX
4. **GitHub Pages Deployment** - Automated CI/CD pipeline
5. **Working Demo** - Ready for submission with <90 second demo video capability

## Technical Stack
```yaml
Frontend:
  - Docusaurus (latest)
  - Custom retro theme (80s terminal aesthetics)
  - Responsive design
  - Interactive code examples

Chatbot:
  - Qwen LLM (via API or local deployment)
  - OpenAI ChatKit SDK (UI framework)
  - Text selection-based queries
  - Conversational AI interface
  - Embedded in book pages

Backend:
  - FastAPI (Python)
  - RAG pipeline implementation
  - Qwen model integration
  - Vector search integration
  - RESTful API endpoints

Databases:
  - Neon Serverless Postgres (user data, chat history)
  - Qdrant Cloud Free Tier (vector embeddings)

Deployment:
  - GitHub Pages (frontend)
  - Backend hosting (Railway/Render/Fly.io)
  - GitHub Actions CI/CD
  - Environment variables management
```

## Content Source
**PRIMARY SOURCE**: `Hackathon_I__Physical_AI___Humanoid_Robotics_Textbook.pdf` located in the project root directory.

The AI agent MUST:
1. Read and parse the PDF content thoroughly
2. Extract all course structure, weekly breakdowns, and learning outcomes
3. Use the PDF as the authoritative source for all technical content
4. Maintain fidelity to the original course design and objectives
5. Expand upon concepts with code examples and practical exercises

## Content Structure (from PDF)

### Module 1: The Robotic Nervous System (ROS 2) - Weeks 1-5
- **Weeks 1-2**: Introduction to Physical AI and embodied intelligence
- **Weeks 3-5**: ROS 2 Fundamentals
- Topics: Nodes, Topics, Services, Actions, URDF, rclpy, Python integration
- Hardware: Understanding sensor systems (LIDAR, cameras, IMUs, force sensors)

### Module 2: The Digital Twin (Gazebo & Unity) - Weeks 6-7
- **Week 6**: Robot Simulation with Gazebo
- **Week 7**: Unity for high-fidelity rendering
- Topics: Physics simulation, URDF/SDF formats, sensor simulation
- Practical: Building simulated environments, collision detection

### Module 3: The AI-Robot Brain (NVIDIA Isaac) - Weeks 8-10
- **Week 8**: NVIDIA Isaac SDK and Isaac Sim introduction
- **Week 9**: AI-powered perception and manipulation
- **Week 10**: Reinforcement learning and sim-to-real transfer
- Topics: Isaac ROS, VSLAM, Nav2 path planning for bipedal movement
- Hardware: Jetson Orin integration, edge computing

### Module 4: Vision-Language-Action (VLA) - Weeks 11-13
- **Weeks 11-12**: Humanoid Robot Development
  - Kinematics and dynamics
  - Bipedal locomotion and balance control
  - Manipulation and grasping
- **Week 13**: Conversational Robotics
  - Voice-to-Action with OpenAI Whisper
  - LLM cognitive planning (translating "Clean the room" to ROS actions)
  - **Capstone Project**: Autonomous Humanoid with voice commands

## Design Requirements

### Retro Theme Specifications
```css
Style Guide:
  - Color Palette: 
    * Primary: Neon green (#00FF41)
    * Secondary: Cyan (#00FFFF), Magenta (#FF00FF)
    * Accent: Amber (#FFB000)
  - Background: 
    * Deep black (#0D0D0D)
    * Dark gray (#1A1A1A)
    * Terminal green tint overlay
  - Fonts: 
    * Headings: JetBrains Mono, Fira Code
    * Body: Source Code Pro, Courier New
    * Monospace for all code blocks
  - Effects: 
    * CRT scanlines (subtle overlay)
    * Text glow/bloom effect
    * Blinking terminal cursor
    * Screen flicker on load
    * Matrix-style text rain (hero section)
  - Accents: 
    * ASCII art robot illustrations
    * Pixel art icons (8-bit style)
    * Terminal-style borders (double-line box drawing)
    * Glitch effects on hover
  - Layout: 
    * Grid-based with terminal-like structure
    * Fixed-width content containers
    * Sidebar with vintage file explorer aesthetic

Components:
  - Landing page with animated ASCII art robot (moving/blinking)
  - Terminal-style navigation menu
  - Code blocks with CRT monitor aesthetic
  - Interactive circuit board diagrams
  - Chatbot in terminal window interface
  - "Boot sequence" loading animations
```

### Landing Page Must Include
1. **Hero Section**:
   - Animated ASCII art robot (walking cycle or waving)
   - Matrix-style falling text effect with robotics keywords
   - Typewriter effect for main heading
   - "System boot" style course introduction
   
2. **Course Overview Section**:
   - 4 module cards with glitch hover effects
   - Terminal-style progress indicators
   - Retro computer icons for each module
   
3. **Hardware Requirements Section**:
   - Styled as a "System Requirements" terminal output
   - Component list with ASCII art diagrams
   - Links to hardware specifications
   
4. **Features Showcase**:
   - Interactive demo of chatbot (click to expand)
   - Code example preview with syntax highlighting
   - Simulation screenshots with CRT filter
   
5. **Call-to-Action**:
   - Retro "Press START to begin" button
   - Enrollment information
   - Course navigation quick links
   
6. **Embedded Chatbot**:
   - Bottom-right corner launcher (styled as terminal icon)
   - Fullscreen terminal-style chat interface
   - Qwen-powered responses with typing animation

## Qwen LLM Integration

### Qwen Model Configuration
```python
# Choose ONE deployment strategy:

# Option 1: Qwen API (recommended for hackathon)
QWEN_API_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_API_KEY = "your-qwen-api-key"
QWEN_MODEL = "qwen-turbo"  # or qwen-plus, qwen-max

# Option 2: Self-hosted Qwen (for local deployment)
QWEN_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # HuggingFace model
QWEN_DEVICE = "cuda"  # or "cpu"
QWEN_MAX_TOKENS = 2048

# Option 3: Ollama (easiest local deployment)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
```

### RAG Pipeline with Qwen
```python
# Embedding Strategy
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI for embeddings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Qwen RAG Prompt Template
RAG_SYSTEM_PROMPT = """You are an expert AI teaching assistant for Physical AI and Humanoid Robotics.
Use the provided context from the textbook to answer questions accurately.
Always cite the specific chapter/section when referencing content.
If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}

Answer based on the context above:"""
```

## RAG Chatbot Architecture

### Features
- **Qwen-powered responses** with technical accuracy
- Answer questions about book content using RAG
- Support text selection queries ("Explain this section")
- Context-aware responses with chapter citations
- Chat history persistence
- Source citations with page/chapter references
- Code example explanations with syntax highlighting
- Multi-turn conversations with memory

### API Endpoints
```python
POST /api/chat/message          # Send message, get Qwen response
POST /api/chat/selection        # Query selected text with context
GET  /api/chat/history          # Retrieve chat history
POST /api/embeddings/generate   # Generate content embeddings from PDF
GET  /api/embeddings/status     # Check embedding generation status
POST /api/qwen/test             # Test Qwen model connection
GET  /api/health                # Health check
```

### Data Flow
1. User sends query via ChatKit interface
2. FastAPI backend receives request
3. Query embedded using OpenAI text-embedding model
4. Qdrant vector search finds relevant PDF content chunks
5. Retrieved context + user query formatted for Qwen
6. Qwen generates response based on RAG context
7. Response streamed back to ChatKit UI with typing effect
8. Interaction logged to Neon Postgres with metadata

### PDF Content Processing Pipeline
```python
# Step 1: Extract text from PDF
- Use PyPDF2 or pdfplumber to extract all text
- Preserve chapter/section structure
- Extract code blocks separately

# Step 2: Intelligent Chunking
- Split by chapters and sections
- Create semantic chunks (512 tokens with 50 overlap)
- Maintain metadata: chapter, week, module, page number

# Step 3: Generate Embeddings
- Use OpenAI text-embedding-3-small for each chunk
- Store in Qdrant with metadata filters

# Step 4: Create Vector Index
- Organize by module/week hierarchy
- Enable filtering by chapter/topic
- Add synonyms and technical terms
```

## Implementation Requirements

### Content Specification:
- Extract complete course structure from PDF
- Map 13 weeks to individual chapter files
- Identify all code examples, diagrams, and exercises from PDF
- Define learning outcomes per chapter (from PDF)
- Create glossary of technical terms
- Hardware requirements documentation
- Assessment rubrics and project guidelines

### Technical Specification:
- Docusaurus v3.x configuration
- Custom retro theme implementation (CSS-in-JS vs CSS modules)
- Required plugins:
  * @docusaurus/plugin-content-docs
  * @docusaurus/plugin-content-blog
  * @docusaurus/plugin-sitemap
  * docusaurus-lunr-search (retro-styled)
- Qwen API integration architecture
- OpenAI ChatKit SDK setup and customization
- FastAPI application structure with async endpoints
- Database schemas for Neon Postgres
- Vector store configuration for Qdrant
- Embedding generation strategy from PDF
- Environment variable management (.env.example)
- CORS and security configurations
- Rate limiting for API endpoints

### UI/UX Specification:
- Component library:
  * RetroButton (3D pressed effect)
  * TerminalBox (bordered container)
  * CodeBlock (CRT monitor style)
  * ChatBubble (terminal message style)
  * LoadingSpinner (ASCII animation)
  * NavBar (file explorer aesthetic)
- Responsive breakpoints:
  * Mobile: 320px - 768px
  * Tablet: 769px - 1024px
  * Desktop: 1025px+
- Typography scale (monospace hierarchy)
- Color system with CSS variables
- Animation specifications (timing, easing)
- Chatbot widget states (minimized, expanded, fullscreen)
- Error state designs
- Loading state patterns

### Deployment Specification:
- GitHub Pages configuration for frontend
- Backend hosting options (Railway preferred)
- GitHub Actions workflows:
  * Build and deploy on push to main
  * Run tests on PR
  * Generate embeddings on content changes
- Environment secrets management
- Domain configuration (optional custom domain)

## Quality Assurance
- All 13 weeks of content extracted from PDF and completed
- Qwen LLM integration fully functional
- RAG pipeline returns accurate, contextual responses
- Retro theme consistently applied across all pages
- Chatbot functional on all pages with terminal aesthetic
- Text selection query feature working
- All API endpoints tested and documented
- Vector embeddings generated for all PDF content
- Responsive design (mobile, tablet, desktop)
- Fast page load times (<2s)
- Chatbot response times (<3s)
- GitHub Pages deployment successful
- Backend deployed and accessible
- README with clear setup instructions
- Demo video created (<90 seconds)
- No console errors or warnings
- Accessibility standards met (WCAG 2.1 AA)
- All content sourced from provided PDF