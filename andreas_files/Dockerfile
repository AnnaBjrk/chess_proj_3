# Dockerfile (for students)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent_interface.py .
COPY game_driver.py .
COPY random_agent.py .
COPY greedy_agent.py .
COPY my_agent.py .

# Set memory limits
RUN ulimit -v 1000000  # 1GB memory limit

# Students will add their my_agent.py later
CMD ["python", "game_driver.py"]