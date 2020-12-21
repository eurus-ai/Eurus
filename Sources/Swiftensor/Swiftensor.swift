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
    
    public var data: [T] {
        get {
            return _storage.data
        }
        set(newData) {
            _storage.data = newData
        }
    }
    
    private var _storage: ArrayStorage<T>
    
    public init(shape: [Int], elements: [T]) {
        precondition(shape.reduce(1, *) == elements.count, "Shape and elements are not compatible.")
        self._shape = shape
        self._storage = ArrayStorage<T>(data: elements)
    }
    
    public func reshaped(_ newShape: [Int]) -> Tensor<T> {
        
        precondition(newShape.filter({ $0 == -1 }).count <= 1, "Invalid shape.")
        
        var newShape = newShape
        if let autoIndex = newShape.firstIndex(of: -1) {
            let prod = -newShape.reduce(1, *)
            newShape[autoIndex] = data.count / prod
        }
        
        return Tensor(shape: newShape, elements: self._storage.data)
    }
    
    public func raveled() -> Tensor<T> {
        return Tensor(shape: [data.count], elements: _storage.data)
    }
    
    public func transposed() -> Tensor<T> {
        let axes: [Int] = (0..<shape.count).reversed()
        return transposed(axes)
    }
    
    public func transposed(_ axes: [Int]) -> Tensor<T> {
        
        precondition(axes.count == shape.count, "Number of `axes` and number of dimensions must correspond.")
        precondition(Set(axes) == Set(0..<shape.count), "Argument `axes` must contain each axis.")
        
        let outShape = axes.map { self.shape[$0] }
        
        let outPointer = UnsafeMutablePointer<T>.allocate(capacity: self.data.count)
        defer { outPointer.deallocate() }
        
        let inIndices = calculateIndices(formatIndicesInAxes(shape, []))
        
        for i in inIndices {
            let oIndex = calculateIndex(outShape, axes.map { i[$0] })
            outPointer.advanced(by: oIndex).pointee = getElement(self, i)
        }
        
        let elements = Array(UnsafeBufferPointer(start: outPointer, count: self.data.count))
        return Tensor(shape: outShape, elements: elements)
    }
}

