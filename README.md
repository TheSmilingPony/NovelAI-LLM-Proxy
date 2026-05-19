Local proxy to handle NovelAI's GLM 4.6 / Xialong model not sending non-streaming responses in a format SillyTavern (and other frontends) can parse. Useful for certain extensions.

Run setup to create the venv, create a .env file using the example with your NAI token, and the run the proxy.

In Silly Tavern create an OpenAI compatible Chat Completion connection connecting to http://0.0.0.0:8001/v1. Select glm-4-6 or xialong-v1 from the models dropdown.

Your SillyTavern should now be able to handle streaming and non-streaming responses from NovelAI.
