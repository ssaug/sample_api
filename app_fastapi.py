#

from fastapi import FastAPI
import asyncio

#application = FastAPI()
application = FastAPI(openapi_url="/openapi.json", docs_url="/docs", redoc_url="/redoc")

# /greeting
@application.get("/greeting")
async def greeting():
    await asyncio.sleep(5)
    return {"message": "Hello World!"}

@application.get("/greeting2")
async def greeting():
    await asyncio.sleep(1)
    return {"message": "Hello World! from API 2 "}

@application.post("/submit")
def submit(data: dict):
    return {"message": "Data submitted successfully!", "data": data["name"]}


## Run in terminal - python  % uvicorn app_fastapi:application --reload

## In any new terminal call conda init and restart terminal to use conda
## C:\anaconda3\Scripts\conda.exe init powershell
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="127.0.0.1", port=8000) ## reload only works in terminal mode execution
