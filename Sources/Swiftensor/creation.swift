import SwiftGD
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
    public static func fromImage(path :String) -> Tensor<Double>? {
        let location = URL(fileURLWithPath: path)
        if let image = Image(url: location) {
            let width = image.size.width
            let height = image.size.height
            var tensor = Tensor<Double>.filled(with: 0, shape: [width,height,3])
            for x in 0 ..< width {
                for y in 0 ..< height {
                    let color = image.get(pixel: Point(x: x, y: y))
                    tensor[x,y,0] = color.redComponent
                    tensor[x,y,1] = color.greenComponent
                    tensor[x,y,2] = color.blueComponent
                }
            }
            return tensor
        } else {
            return nil
        }
    }
}

