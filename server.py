# Two server routes that OctoAI containers should have:
# a route for inference requests (e.g. ”/predict”). This route for inference requests must receive JSON inputs and JSON outputs.
# a route for health checks (e.g. ”/healthcheck”).
# Number of workers (not required). Typical best practice is to make this number some function of the # of CPU cores that the server has access to and should use.

"""HTTP Inference serving interface using sanic."""
import os

from custommodel import CustomModel
from sanic import Request, Sanic, response

_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

# Load and initialize the model on startup globally, so it can be reused.
model_instance = CustomModel(ckpt='models/distil-240716-model_epoch_9.pth')
"""Global instance of the model to serve."""

server = Sanic("server")
"""Global instance of the web server."""


@server.route("/healthcheck", methods=["GET"])
def healthcheck(_: Request) -> response.JSONResponse:
    """Responds to healthcheck requests.

    :param request: the incoming healthcheck request.
    :return: json responding to the healthcheck.
    """
    return response.json({"healthy": "yes"})


@server.route("/reconstruction/predict", methods=["POST"])
def predict(request: Request) -> response.JSONResponse:
    """Responds to inference/prediction requests.

    :param request: the incoming request containing inputs for the model.
    :return: json containing the inference results.
    """

    try:
        inputs = request.json
        output = model_instance.predict(inputs)
        return response.json(output)
    except Exception as e:
        return response.json({'error': str(e)}, status=500)


def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    server.run(host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    main()
