


# ACUD Project – Run Instructions

## 1. Activate the Python Environment

Open PowerShell and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "C:\Users\Dell\Downloads\capital_complaints_ui (1)\.venv\Scripts\Activate.ps1"
```

Make sure the virtual environment is activated before starting the backend or Python HTTP servers.

---

## 2. Terminal 1 – Backend API Server

Navigate to the backend directory:

```powershell
cd "C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype\backend"
```

Start the unified FastAPI application:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Backend URLs:**

- Agents Dashboard: http://127.0.0.1:8000/agents-ui
- API Documentation: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

Keep this terminal running.

---

## 3. Terminal 2 – Frontend Development Server

Navigate to the frontend project directory (the directory containing `package.json`).

Start the frontend:

```powershell
npm run dev
```

**Frontend URL:**

http://127.0.0.1:5173/

Keep this terminal running.

---

## 4. Terminal 3 – Residential Sales and Executive Dashboards

Navigate to the project root:

```powershell
cd "C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype"
```

Start the static file server on port 8081:

```powershell
python -m http.server 8081
```

**Dashboard URLs:**

- Residential Sales Map: http://127.0.0.1:8081/residential-sales-map.html
- ACUD Executive Dashboard: http://127.0.0.1:8081/acud-executive-dashboard.html

Keep this terminal running.

---

## 5. Terminal 4 – Dashboard 2

Navigate to the project root:

```powershell
cd "C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype"
```

Start the static file server on port 8080:

```powershell
python -m http.server 8080
```

**Dashboard URL:**

http://localhost:8080/public/dashboard2.html#lab

Keep this terminal running.

---

## 6. Running Services Summary

| Service | Port | URL |
|---|---:|---|
| FastAPI Backend | 8000 | http://127.0.0.1:8000 |
| Frontend Dev Server | 5173 | http://127.0.0.1:5173 |
| Residential / Executive Dashboards | 8081 | http://127.0.0.1:8081 |
| Dashboard 2 | 8080 | http://localhost:8080 |

## 7. Important Notes

- Run each service in its own terminal.
- Keep all four terminals open while using the project.
- Start the backend before testing dashboards that depend on its API.
- Ports 8000, 5173, 8080, and 8081 must be available.
- The backend is the unified FastAPI application. Do not start the old `uvicorn api:app` server separately.
- If a port is already occupied, identify the process using it before starting another server.
- To stop a server, press `Ctrl + C` in its terminal.

## 8. Recommended Startup Order

1. Activate the Python environment.
2. Start the FastAPI backend (Terminal 1).
3. Start the frontend (Terminal 2).
4. Start the static server on port 8081 (Terminal 3).
5. Start the static server on port 8080 (Terminal 4).
6. Open the required dashboard URLs in your browser.
Steps to run 

Activeate env.

mine: (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "c:\Users\Dell\Downloads\capital_complaints_ui (1)\.venv\Scripts\Activate.ps1")

 terminal 1: backend server:
PS C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype\backend> uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

http://127.0.0.1:8000/agents-ui
http://localhost:8080/public/dashboard2.html#lab


 terminal 2: front end : npm run dev 
http://localhost:5173/


terminal 3 : PS C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype> python -m http.server 8081

http://127.0.0.1:8081/residential-sales-map.html

http://127.0.0.1:8081/acud-executive-dashboard.html

terminal 4 : PS C:\Users\Dell\Downloads\acud-prototype (1)\acud-prototype> python -m http.server 8080
http://localhost:8080/public/dashboard2.html#lab