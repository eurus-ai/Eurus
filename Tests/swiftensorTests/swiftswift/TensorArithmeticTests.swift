import XCTest
@testable import Swiftensor

class TensorArithmeticTests: XCTestCase {
    
    func testUnaryPlus() {
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(unaryPlus(a), a)
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(unaryPlus(a), a)
        }
    }
    
    func testUnaryPlusOperator() {
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(+a, a)
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(+a, a)
        }
    }
    
    func testUnaryMinus() {
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(unaryMinus(a), Tensor(shape: [2, 2], elements: [-1, -2, -3, -4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(unaryMinus(a),
                                       Tensor(shape: [2, 2], elements: [-1, -2, -3, -4]))
        }
    }
    
    func testUnaryMinusOperator() {
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(-a, Tensor(shape: [2, 2], elements: [-1, -2, -3, -4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(-a,
                                       Tensor(shape: [2, 2], elements: [-1, -2, -3, -4]))
        }
    }
    
    func testAdd() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            
            XCTAssertEqual(Swiftensor.add(a, 1), Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
            XCTAssertEqual(Swiftensor.add(1, a), Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(Swiftensor.add(a, 1),
                                       Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
            XCTAssertEqual(Swiftensor.add(1, a),
                                       Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(Swiftensor.add(a, b), Tensor(shape: [2, 2], elements: [0, 2, 4, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(Swiftensor.add(a, b), Tensor(shape: [2, 2], elements: [0, 2, 4, 4]))
        }
    }
    
    func testAddOperator() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a + 1, Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
            XCTAssertEqual(1 + a, Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a + 1,
                                       Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
            XCTAssertEqual(1 + a,
                                       Tensor(shape: [2, 2], elements: [2, 3, 4, 5]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(a + b, Tensor(shape: [2, 2], elements: [0, 2, 4, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(a + b, Tensor(shape: [2, 2], elements: [0, 2, 4, 4]))
        }
    }
    
    func testSubtract() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(subtract(a, 1), Tensor(shape: [2, 2], elements: [0, 1, 2, 3]))
            XCTAssertEqual(subtract(1, a), Tensor(shape: [2, 2], elements: [0, -1, -2, -3]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(subtract(a, 1), Tensor(shape: [2, 2], elements: [0, 1, 2, 3]))
            XCTAssertEqual(subtract(1, a), Tensor(shape: [2, 2], elements: [0, -1, -2, -3]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(subtract(a, b), Tensor(shape: [2, 2], elements: [2, 2, 2, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(subtract(a, b), Tensor(shape: [2, 2], elements: [2, 2, 2, 4]))
        }
    }
    
    func testSubtractOperator() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a - 1, Tensor(shape: [2, 2], elements: [0, 1, 2, 3]))
            XCTAssertEqual(1 - a, Tensor(shape: [2, 2], elements: [0, -1, -2, -3]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a - 1, Tensor(shape: [2, 2], elements: [0, 1, 2, 3]))
            XCTAssertEqual(1 - a, Tensor(shape: [2, 2], elements: [0, -1, -2, -3]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(a - b, Tensor(shape: [2, 2], elements: [2, 2, 2, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 1, 0])
            XCTAssertEqual(a - b, Tensor(shape: [2, 2], elements: [2, 2, 2, 4]))
        }
    }
    
    func testMultiply() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(multiply(a, 2), Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
            XCTAssertEqual(multiply(2, a), Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(multiply(a, 2), Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
            XCTAssertEqual(multiply(2, a), Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 2, 1])
            XCTAssertEqual(multiply(a, b), Tensor(shape: [2, 2], elements: [-1, 0, 6, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 2, 1])
            XCTAssertEqual(multiply(a, b), Tensor(shape: [2, 2], elements: [-1, 0, 6, 4]))
        }
    }
    
    func testMultiplyOperator() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a * 2, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
            XCTAssertEqual(2 * a, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a * 2, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
            XCTAssertEqual(2 * a, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 0, 2, 1])
            XCTAssertEqual(a * b, Tensor(shape: [2, 2], elements: [-1, 0, 6, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 0, 2, 1])
            XCTAssertEqual(a * b, Tensor(shape: [2, 2], elements: [-1, 0, 6, 4]))
        }
    }
    
    func testDivide() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(divide(a, 2), Tensor(shape: [2, 2], elements: [0, 1, 1, 2]))
            XCTAssertEqual(divide(2, a), Tensor(shape: [2, 2], elements: [2, 1, 0, 0]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(divide(a, 2), Tensor(shape: [2, 2], elements: [0.5, 1.0, 1.5, 2.0]))
            XCTAssertEqual(divide(2, a), Tensor(shape: [2, 2], elements: [2.0, 1.0, 2.0/3.0, 0.5]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 2, 6, 1])
            XCTAssertEqual(divide(a, b), Tensor(shape: [2, 2], elements: [-1, 1, 0, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 2, 6, 1])
            XCTAssertEqual(divide(a, b), Tensor(shape: [2, 2], elements: [-1, 1, 0.5, 4]))
        }
    }
    
    func testDivideOperator() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a / 2, Tensor(shape: [2, 2], elements: [0, 1, 1, 2]))
            XCTAssertEqual(2 / a, Tensor(shape: [2, 2], elements: [2, 1, 0, 0]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a / 2, Tensor(shape: [2, 2], elements: [0.5, 1.0, 1.5, 2.0]))
            XCTAssertEqual(2 / a, Tensor(shape: [2, 2], elements: [2.0, 1.0, 2.0/3.0, 0.5]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [-1, 2, 6, 1])
            XCTAssertEqual(a / b, Tensor(shape: [2, 2], elements: [-1, 1, 0, 4]))
        }
        do {
            let a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [-1, 2, 6, 1])
            XCTAssertEqual(a / b, Tensor(shape: [2, 2], elements: [-1, 1, 0.5, 4]))
        }
    }
    
    func testModulo() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(modulo(a, 2), Tensor(shape: [2, 2], elements: [1, 0, 1, 0]))
            XCTAssertEqual(modulo(2, a), Tensor(shape: [2, 2], elements: [0, 0, 2, 2]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [3, 2, 2, 3])
            XCTAssertEqual(modulo(a, b), Tensor(shape: [2, 2], elements: [1, 0, 1, 1]))
        }
    }
    
    func testModuloOperator() {
        // Tensor and scalar
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            XCTAssertEqual(a % 2, Tensor(shape: [2, 2], elements: [1, 0, 1, 0]))
            XCTAssertEqual(2 % a, Tensor(shape: [2, 2], elements: [0, 0, 2, 2]))
        }
        
        // Tensor and Tensor
        do {
            let a = Tensor<Int>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Int>(shape: [2, 2], elements: [3, 2, 2, 3])
            XCTAssertEqual(a % b, Tensor(shape: [2, 2], elements: [1, 0, 1, 1]))
        }
    }
    
    func testAddAssign() {
        // scalar
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            a += 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [3, 4, 5, 6]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            a += 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [3, 4, 5, 6]))
        }
        // Tensor
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor(shape: [2, 2], elements: [1, -2, 6, 0])
            a += b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [2, 0, 9, 4]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [1, -2, 6, 0])
            a += b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [2, 0, 9, 4]))
        }
    }
    
    func testSubtractAssign() {
        // scalar
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            a -= 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [-1, 0, 1, 2]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            a -= 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [-1, 0, 1, 2]))
        }
        // Tensor
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor(shape: [2, 2], elements: [1, -2, 6, 0])
            a -= b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [0, 4, -3, 4]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [1, -2, 6, 0])
            a -= b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [0, 4, -3, 4]))
        }
    }
    
    func testMultiplyAssign() {
        // scalar
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            a *= 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            a *= 2
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [2, 4, 6, 8]))
        }
        // Tensor
        do {
            var a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor(shape: [2, 2], elements: [1, -2, 6, 0])
            a *= b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [1, -4, 18, 0]))
        }
        do {
            var a = Tensor<Double>(shape: [2, 2], elements: [1, 2, 3, 4])
            let b = Tensor<Double>(shape: [2, 2], elements: [1, -2, 6, 0])
            a *= b
            XCTAssertEqual(a, Tensor(shape: [2, 2], elements: [1, -4, 18, 0]))
        }
    }
    
    
    static var allTests: [(String, (TensorArithmeticTests) -> () throws -> Void)] {
        return [
            ("testUnaryPlus", testUnaryPlus),
            ("testUnaryPlusOperator", testUnaryPlusOperator),
            ("testUnaryMinus", testUnaryMinus),
            ("testUnaryMinusOperator", testUnaryMinusOperator),
            ("testAdd", testAdd),
            ("testAddOperator", testAddOperator),
            ("testSubtract", testSubtract),
            ("testSubtractOperator", testSubtractOperator),
            ("testMultiply", testMultiply),
            ("testMultiplyOperator", testMultiplyOperator),
            ("testDivide", testDivide),
            ("testDivideOperator", testDivideOperator),
            ("testModulo", testModulo),
            ("testModuloOperator", testModuloOperator),
            ("testAddAssign",testAddAssign),
            ("testSubtractAssign",testSubtractAssign),
            ("testMultiplyAssign",testMultiplyAssign)
        ]
    }
}
