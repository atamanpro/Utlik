from fastapi import FastAPI
from integrations.router import router


app = FastAPI(
    title="Integrations API",
    description="""
    You can be able to do the following:
    **first_endpoint** - get some data from ..
    **second_endpoint** - post some data to ..
    """,
    version="0.0.1",
    docs_url="/integrations/docs",
    redoc_url="/integrations/redoc",
    openapi_url="/integrations/openapi.json")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=81, reload=True)