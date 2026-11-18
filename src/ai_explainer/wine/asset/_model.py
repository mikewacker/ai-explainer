from ai_explainer.wine.world import WineTrace


def latex_prediction_hand(trace: WineTrace) -> list[str]:
    """Get the steps to compute a prediction for the hand-crafted model."""
    return [
        r"\hat{y}",
        f"{trace.w[0]:g} x_1 + {trace.w[1]:g} x_2",
        rf"{trace.w[0]:g} \cdot {trace.x[0]:g} + {trace.w[1]:g} \cdot {trace.x[1]:g}",
        f"{trace.w[0] * trace.x[0]:g} + {trace.w[1] * trace.x[1]:g}",
        f"{trace.y_hat:g}",
    ]


def latex_prediction_no_params(trace: WineTrace) -> list[str]:
    """Get the partial steps to compute a prediction for the AI model with unset parameters."""
    return [
        r"\hat{y}",
        "w_1 x_1 + w_2 x_2",
        rf"w_1 \cdot {trace.x[0]:g} + w_2 \cdot {trace.x[1]:g}",
    ]
