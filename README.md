# Clip-Env

This repository contains a Python project that uses the CLIP model to analyze images and infer descriptive phrases. The project processes images in a specified directory and generates JSON files with analysis results.

## Setup Instructions

### Prerequisites

- Python 3.12
- Git

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/VTAkbay/clip-env.git
cd clip-env
```

Create and Activate a Virtual Environment
Create a virtual environment named venv and activate it:

On macOS and Linux

```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install Dependencies
Install the required dependencies using pip:

```bash
Copy code
pip install -r requirements.txt
```

Run the Project
To run the script and process images in a specified directory, use the following command:

```bash
Copy code
python clip_analyze.py <directory_path>
```
