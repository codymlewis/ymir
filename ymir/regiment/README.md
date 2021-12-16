# Regiment
Endpoint side functionalities, inclusive of [adversaries](/ymir/regiment/adversaries).
The [Scout](/ymir/regiment/scout#Scout) object is the basic/standard endpoint
collaborator for federated learning, while the adversary modules each act either as modifiers for [Scout](/ymir/regiment/scout#Scout)
instances (with the function `convert`) or as update transforms at the network level (with the `GradientTransform` objects).