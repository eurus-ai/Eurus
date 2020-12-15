import Foundation

// MARK: - Normal
public func sqrt<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _sqrt(arg)
}

public func exp<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _exp(arg)
}

public func log<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _log(arg)
}

public func sin<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _sin(arg)
}

public func cos<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _cos(arg)
}

public func tan<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return _tan(arg)
}

func _sqrt<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.sqrt)
}

func _exp<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.exp)
}

func _log<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.log)
}

func _sin<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.sin)
}

func _cos<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.cos)
}

func _tan<T: FloatingPointFunctions>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, T.tan)
}
