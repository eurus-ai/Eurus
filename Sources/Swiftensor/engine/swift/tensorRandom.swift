extension Tensor where T: FloatingPointFunctions & FloatingPoint {
    
    public static func uniform(low: T = 0, high: T = 1, shape: [Int]) -> Tensor<T> {
        let count = shape.reduce(1, *)
        let elements = (0..<count).map { _ in _uniform(low: low, high: high) }
        
        return Tensor<T>(shape: shape, elements: elements)
    }
}

extension Tensor where T: FloatingPoint & FloatingPointFunctions & Arithmetic {
    
    public static func normal(mu: T = 0, sigma: T = 1, shape: [Int]) -> Tensor<T> {
        let u1 = uniform(low: T(0), high: T(1), shape: shape)
        let u2 = uniform(low: T(0), high: T(1), shape: shape)
        
        let stdNormal =  sqrt(-2*log(u1)) * cos(2*T.pi*u2)
        
        return stdNormal*sigma + mu
    }
}
