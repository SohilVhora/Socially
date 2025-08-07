from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    print("got msg")
    return {"Message":"Hello World"}
