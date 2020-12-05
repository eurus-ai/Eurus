extension Tensor {
    public static func filled(with value: T, shape: [Int]) -> Tensor<T> {
        let elements = [T](repeating: value, count: shape.reduce(1, *))
        return Tensor(shape: shape, elements: elements)
    }
}

extension Tensor where T: ZeroOne {
    public static func zeros(_ shape: [Int]) -> Tensor<T> {
        return Tensor.filled(with: T.zero, shape: shape)
    }
    
    public static func ones(_ shape: [Int]) -> Tensor<T> {
        return Tensor.filled(with: T.one, shape: shape)
    }
    
    public static func eye(_ size: Int) -> Tensor<T> {
        var identity = zeros([size, size])
        (0..<size).forEach { identity[$0, $0] = T.one }
        return identity
    }
}

extension Tensor where T: Strideable {
    public static func range(from: T, to: T, stride: T.Stride) -> Tensor<T> {
        let elements = Array(Swift.stride(from: from, to: to, by: stride))
        return Tensor(shape: [elements.count], elements: elements)
    }
}

extension Tensor where T: FloatingPoint {
    public static func linspace(low: T, high: T, count: Int) -> Tensor<T> {
        let d = high - low
        let steps = T(count-1)
        let elements = (0..<count).map { low + T($0)*d/steps }
        return Tensor(shape: [count], elements: elements)
    }
}
