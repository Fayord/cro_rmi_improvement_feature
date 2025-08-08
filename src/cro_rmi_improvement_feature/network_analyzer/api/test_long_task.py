import sqlite3
import time
import uuid
import json
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import random
import uvicorn

# In a real-world scenario, you might use a more robust
# ORM like SQLAlchemy, but for a simple case, sqlite3 is fine.
DATABASE_URL = "tasks.db"


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn


def setup_database():
    """Initializes the database and creates the tasks table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            result TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# Call this function to ensure the database is ready when the app starts
setup_database()

app = FastAPI()


class TaskData(BaseModel):
    data: Dict[str, Any]


def long_running_task(task_id: str, data: Dict[str, Any]):
    """
    Simulates a long-running process and updates the task status in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Simulate processing time
        # make it random fail

        time.sleep(random.randint(1, 10))  # Adjust this for your actual workload

        if random.random() < 0.5:
            raise Exception("Random failure")
        # Simulate a successful result
        result_data = {"processed_data": f"Processed: {data['data']}"}

        cursor.execute(
            "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
            ("success", json.dumps(result_data), task_id),
        )
        conn.commit()

    except Exception as e:
        # Handle potential errors during processing
        error_data = {"error": str(e)}

        cursor.execute(
            "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
            ("fail", json.dumps(error_data), task_id),
        )
        conn.commit()

    finally:
        conn.close()


@app.post("/api/process")
async def start_processing(task_data: TaskData, background_tasks: BackgroundTasks):
    """
    Accepts data, starts a background task, and returns a task ID.
    """
    task_id = str(uuid.uuid4())

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (task_id, status) VALUES (?, ?)", (task_id, "in_progress")
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(long_running_task, task_id, task_data.dict())

    return {"message": "Processing started", "task_id": task_id}


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """
    Retrieves the status and result of a background task using its ID.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
    task = cursor.fetchone()
    conn.close()

    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")

    status = task["status"]
    result = task["result"]

    if status == "success":
        return {"status": status, "data": json.loads(result)}
    elif status == "fail":
        return {"status": status, "error": json.loads(result)}
    else:  # status == "in_progress"
        return {"status": status, "message": "Processing is still in progress"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
