public protocol Arithmetic {
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    
    static func += (lhs: inout Self, rhs: Self)
    static func -= (lhs: inout Self, rhs: Self)
    static func *= (lhs: inout Self, rhs: Self)
    static func /= (lhs: inout Self, rhs: Self)
    
    init<T>(_ value:T) where T: BinaryFloatingPoint
    init<T>(_ value:T) where T: BinaryInteger
    
}

public protocol Moduloable {
    static func % (lhs: Self, rhs: Self) -> Self
}

//extension Numeric: Arithmetic{}
//extension BinaryInteger: Moduloable{}

extension Int: Arithmetic, Moduloable {}
extension Int32: Arithmetic, Moduloable {}
extension UInt: Arithmetic, Moduloable {}
extension UInt8: Arithmetic, Moduloable {}
extension Float: Arithmetic {}
extension Double: Arithmetic {}

//
//public extension Int {
//    init<T: Arithmetic>(_ element: T) {
//        self = Int(element)
//    }
//}
//
//public extension Int32 {
//    init<T: Arithmetic>(_ element: T) {
//        self = Int32(element)
//    }
//}
//
//public extension UInt {
//    init<T: Arithmetic>(_ element: T) {
//        self = UInt(element)
//    }
//}
//
//public extension UInt8 {
//    init<T: Arithmetic>(_ element: T) {
//        self = UInt8(element)
//    }
//}
//
//public extension Float {
//    init<T: Arithmetic>(_ element: T) {
//        self = Float(element)
//    }
//}
//
//
//public extension Double {
//    init<T: Arithmetic>(_ element: T) {
//        self = Double(element)
//    }
//}
