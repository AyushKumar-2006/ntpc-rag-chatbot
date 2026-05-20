## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/AyushKumar-2006/ntpc-rag-chatbot.git
   cd ntpc-rag-chatbot
```

2. **Set up the backend**
```bash
   cd backend
   pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
   cp .env.example .env
   # Add your API keys to .env
```

4. **Set up the frontend**
```bash
   cd frontend
   npm install
```

### Running the App

1. **Start the backend**
```bash
   cd backend
   python main.py
```

2. **Start the frontend**
```bash
   cd frontend
   npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

Type any question about NTPC in the chat interface, for example:
- *"What was NTPC's revenue in 2023-24?"*
- *"How many power plants does NTPC operate?"*
- *"What are NTPC's renewable energy targets?"*

## Data Sources

Annual reports included:
- NTPC Annual Report 2023-24
- NTPC Annual Report 2024-25
- Annual Reports from 2017-2022

## License

MIT License