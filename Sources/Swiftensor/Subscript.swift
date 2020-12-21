extension Tensor {
    public subscript(index: Int...) -> T {
        get {
            return getElement(self, index)
        }
        set {
            setElement(&self, index, newValue: newValue)
        }
    }
    
    public subscript(indices: [Int]?...) -> Tensor<T> {
        get {
            return getSubarray(self, indices)
        }
        set {
            setSubarray(&self, indices, newValue)
        }
    }
    
    public subscript(ranges: CountableRange<Int>?...) -> Tensor<T> {
        let indices = ranges.map { range in
            range.map { r in
                [Int](r)
            }
        }
        return getSubarray(self, indices)
    }
    
    public subscript(ranges: CountableClosedRange<Int>?...) -> Tensor<T> {
        let indices = ranges.map { range in
            range.map { r in
                [Int](r)
            }
        }
        return getSubarray(self, indices)
    }
}

func getElement<T>(_ array: Tensor<T>, _ index: [Int]) -> T {
    let elementIndex = calculateIndex(array.shape, index)
    return array.data[elementIndex]
}

func setElement<T>(_ array: inout Tensor<T>, _ index: [Int], newValue: T) {
    let elementIndex = calculateIndex(array.shape, index)
    array.data[elementIndex] = newValue
}

func getSubarray<T>(_ array: Tensor<T>, _ indices: [[Int]?]) -> Tensor<T> {
    
    let indices = formatIndicesInAxes(array.shape, indices)
    
    let shape = indices.map { $0.count }
    let elements = calculateIndices(indices).map { getElement(array, $0) }
    
    return Tensor(shape: shape, elements: elements)
}

func setSubarray<T>(_ array: inout Tensor<T>, _ indices: [[Int]?], _ newValue: Tensor<T>) {
    let indices = formatIndicesInAxes(array.shape, indices)
    
    let shape = indices.map { $0.count }
    precondition(shape == newValue.shape, "Arguments are incompatible.")
    
    for (i, index) in calculateIndices(indices).enumerated() {
        setElement(&array, index, newValue: newValue.data[i])
    }
}
