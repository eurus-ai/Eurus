// MARK: - Normal

// MARK: Extensions
extension Tensor where T: Comparable {
    
    public func min() -> T {
        return _min(self)
    }
    
    public func max() -> T {
        return _max(self)
    }
    
    public func min(along axis: Int) -> Tensor<T> {
        return _min(self, along: axis)
    }
    
    public func max(along axis: Int) -> Tensor<T> {
        return _max(self, along: axis)
    }
    
}

extension Tensor where T: Arithmetic {
    
    public func sum() -> T {
        return _sum(self)
    }
    
    public func sum(along axis: Int) -> Tensor<T> {
        return _sum(self, along: axis)
    }
}

extension Tensor where T: Arithmetic & FloatingPoint {
    
    public func mean() -> T {
        return _mean(self)
    }
    
    public func mean(along axis: Int) -> Tensor<T> {
        return _mean(self, along: axis)
    }
}

// MARK: Whole elements

func _min<T: Comparable>(_ arg: Tensor<T>) -> T {
    return arg.data.min()!
}

func _max<T: Comparable>(_ arg: Tensor<T>) -> T {
    return arg.data.max()!
}

func _sum<T: Arithmetic>(_ arg: Tensor<T>) -> T {
    let initial = arg.data.first!
    return arg.data.dropFirst().reduce(initial, +)
}

func _mean<T: Arithmetic & FloatingPoint>(_ arg: Tensor<T>) -> T {
    return _sum(arg) / T(arg.data.count)
}

// MARK: Along axis

private func reduce<T>(_ arg: Tensor<T>, along axis: Int, handler: (T, T) -> T) -> Tensor<T> {
    var axis = axis
    if axis < 0 {
        axis += arg.shape.count
    }
    
    precondition(0 <= axis && axis < arg.shape.count, "Invalid axis.")
    
    let outShape = arg.shape.removed(at: axis)
    let count = arg.data.count / arg.shape[axis]
    
    let outPointer = UnsafeMutablePointer<T>.allocate(capacity: count)
    defer { outPointer.deallocate() }
    
    let majorSize = arg.shape.prefix(upTo: axis).reduce(1, *)
    let minorSize = arg.shape.dropFirst(axis+1).reduce(1, *)
    
    var op = outPointer
    for major in 0..<majorSize {
        for minor in 0..<minorSize {
            var ip = UnsafePointer(arg.data)
            // init
            ip += major*minorSize*arg.shape[axis] + minor
            op.pointee = ip.pointee
            // reduce rest
            ip += minorSize
            for _ in 1..<arg.shape[axis] {
                op.pointee = handler(ip.pointee, op.pointee)
                ip += minorSize
            }
            op += 1
        }
    }
    
    return Tensor(shape: outShape,
                   elements: Array(UnsafeBufferPointer(start: outPointer, count: count)))
}

func _min<T: Comparable>(_ arg: Tensor<T>, along axis: Int) -> Tensor<T> {
    return reduce(arg, along: axis, handler: min)
}

func _max<T: Comparable>(_ arg: Tensor<T>, along axis: Int) -> Tensor<T> {
    return reduce(arg, along: axis, handler: max)
}

func _sum<T: Arithmetic>(_ arg: Tensor<T>, along axis: Int) -> Tensor<T> {
    return reduce(arg, along: axis, handler: +)
}

func _mean<T: Arithmetic & FloatingPoint>(_ arg: Tensor<T>, along axis: Int) -> Tensor<T> {
    return _sum(arg, along: axis) / T(arg.shape[axis])
}
