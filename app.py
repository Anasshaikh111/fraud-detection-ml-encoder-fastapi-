from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

from evaluator import evaluate_transaction

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------- HOME PAGE ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ---------------- SINGLE PREDICTION ----------------
@app.post("/predict_single", response_class=HTMLResponse)
async def predict_single(
    request: Request,
    amount: float = Form(...),
    time: float = Form(...),
    description: str = Form("")
):

    # auto-generate 28 PCA numeric features
    features = np.random.normal(0, 1, 28).tolist()

    # add time and amount
    features.append(time)
    features.append(amount)

    # call evaluator
    result = evaluate_transaction(features, description)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "single_result": result
        }
    )


# ---------------- BULK CSV UPLOAD ----------------
@app.post("/predict_file", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    results = []
    for _, row in df.iterrows():
        features = row[:-1].tolist()
        description = ""  # not using encoder for CSV yet
        res = evaluate_transaction(features, description)
        results.append(res)

    fraud = sum(1 for r in results if r["prediction"] == "Fraudulent")
    legit = len(results) - fraud

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "batch_results": results,
            "fraud": fraud,
            "legit": legit,
        }
    )
