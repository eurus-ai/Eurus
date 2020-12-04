public struct Tensor<T> {
    private var _shape: [Int]
    public var shape: [Int] {
        get {
            return _shape
        }
        set {
            self = reshaped(newValue)
        }
    }
    
    public internal(set) var storage: ArrayStorage<T>
    
    public init(shape: [Int], elements: [T]) {
        precondition(shape.reduce(1, *) == elements.count, "Shape and elements are not compatible.")
        self._shape = shape
        self.storage = ArrayStorage<T>(data: elements)
    }
    
    public func reshaped(_ newShape: [Int]) -> Tensor<T> {
        
        precondition(newShape.filter({ $0 == -1 }).count <= 1, "Invalid shape.")
        
        var newShape = newShape
        if let autoIndex = newShape.firstIndex(of: -1) {
            let prod = -newShape.reduce(1, *)
            newShape[autoIndex] = storage.data.count / prod
        }
        
        return Tensor(shape: newShape, elements: self.storage.data)
    }
    
    public func raveled() -> Tensor<T> {
        return Tensor(shape: [storage.data.count], elements: storage.data)
    }
}

