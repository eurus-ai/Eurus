import SwiftCV
import Foundation

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

extension Tensor  {

    
    public static func fromImage(path :String) -> Tensor<UInt8>? {
        var cvImg = imread(path)
        cvImg = cvtColor(cvImg, nil, ColorConversionCode.COLOR_BGR2RGB)
        let ptr = cvImg.dataPtr
        let matShape: [Int] = [cvImg.rows, cvImg.cols, cvImg.channels]
        let length = cvImg.rows*cvImg.cols*cvImg.channels
        let uint8Ptr = ptr.bindMemory(to: UInt8.self, capacity: length)
        let uint8Buffer = UnsafeBufferPointer(start: uint8Ptr, count: length)
        let output = Array(uint8Buffer)
        let tensor = Tensor<UInt8>(shape: matShape, elements: output).transposed([2,0,1])
        return tensor
        
    }
}


//MARK: Tensor - array conversion
public extension Tensor {
    
    /// Creates a tensor value holding the provided scalar. The tensor will have an empty shape.
    /// - Parameters:
    ///   - e: Element
    init(_ e: T) {
        self.init(shape: [],elements:[e])
    }
    
    init(_ v: [T]) {
        self.init(shape: [v.count],elements: v)
    }
    
    init(_ v: [[T]]) {
        self.init(shape: [v.count, v.first?.count ?? 0],elements: Array(v.joined()))
    }
    

    init(_ v: [[[T]]]) {
        self.init(
            shape: [v.count, v.first?.count ?? 0, v.first?.first?.count ?? 0],
            elements: Array(v.joined().joined())
        )
    }
    
    init(_ v: [[[[T]]]]) {
        self.init(
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0
            ],
            elements: Array(v.joined().joined().joined())
        )
    }
    
    init(_ v: [[[[[T]]]]]) {
        self.init(
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0,
                v.first?.first?.first?.first?.count ?? 0
            ],
            elements: Array(v.joined().joined().joined().joined())
        )
    }
}

