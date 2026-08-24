# Grockle

An agent to make planning easier

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate the virtual enviroment

MacOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installing dependencies

```bash
pip install -r requirements.txt
```

### 4. Setting up ENV

Create a .env file in the project root and set up `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### 5. Run the app

```bash
chainlit run app.py
```
