import Foundation
import XCTest
@testable import Swiftensor

class TensorFloatingPointFunctionsTests: XCTestCase {
    
    func testSqrt() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(sqrt(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sqrt)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_sqrt(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sqrt)))
        }
    }
    
    func testExp() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(exp(a),
                                       Tensor(shape: [2, 2], elements: elements.map(exp)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(exp(a),
                                       Tensor(shape: [2, 2], elements: elements.map(exp)))
        }
    }
    
    func testLog() {
        do {
            let elements: [Float] = [0.1, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(log(a),
                                       Tensor(shape: [2, 2], elements: elements.map(log)))
        }
        do {
            let elements: [Double] = [0.1, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(log(a),
                                       Tensor(shape: [2, 2], elements: elements.map(log)))
        }
    }
    
    func testSin() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(sin(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sin)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(sin(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sin)))
        }
    }
    
    func testCos() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(cos(a),
                                       Tensor(shape: [2, 2], elements: elements.map(cos)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(cos(a),
                                       Tensor(shape: [2, 2], elements: elements.map(cos)))
        }
    }
    
    func testTan() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(tan(a),
                                       Tensor(shape: [2, 2], elements: elements.map(tan)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(tan(a),
                                       Tensor(shape: [2, 2], elements: elements.map(tan)))
        }
    }
    
    func testSqrtNormal() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_sqrt(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sqrt)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_sqrt(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sqrt)))
        }
    }
    
    func testExpNormal() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_exp(a),
                                       Tensor(shape: [2, 2], elements: elements.map(exp)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_exp(a),
                                       Tensor(shape: [2, 2], elements: elements.map(exp)))
        }
    }
    
    func testLogNormal() {
        do {
            let elements: [Float] = [0.1, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_log(a),
                                       Tensor(shape: [2, 2], elements: elements.map(log)))
        }
        do {
            let elements: [Double] = [0.1, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_log(a),
                                       Tensor(shape: [2, 2], elements: elements.map(log)))
        }
    }
    
    func testSinNormal() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_sin(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sin)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_sin(a),
                                       Tensor(shape: [2, 2], elements: elements.map(sin)))
        }
    }
    
    func testCosNormal() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_cos(a),
                                       Tensor(shape: [2, 2], elements: elements.map(cos)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_cos(a),
                                       Tensor(shape: [2, 2], elements: elements.map(cos)))
        }
    }
    
    func testTanNormal() {
        do {
            let elements: [Float] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_tan(a),
                                       Tensor(shape: [2, 2], elements: elements.map(tan)))
        }
        do {
            let elements: [Double] = [0, 0.5, 1, 1.5]
            let a = Tensor(shape: [2, 2], elements: elements)
            XCTAssertEqual(_tan(a),
                                       Tensor(shape: [2, 2], elements: elements.map(tan)))
        }
    }
    
    static var allTests: [(String, (TensorFloatingPointFunctionsTests) -> () throws -> Void)] {
        return [
            ("testSqrt", testSqrt),
            ("testExp", testExp),
            ("testLog", testLog),
            ("testSin", testSin),
            ("testCos", testCos),
            ("testTan", testTan),
            ("testSqrtNormal", testSqrtNormal),
            ("testExpNormal", testExpNormal),
            ("testLogNormal", testLogNormal),
            ("testSinNormal", testSinNormal),
            ("testCosNormal", testCosNormal),
            ("testTanNormal", testTanNormal)
        ]
    }
    
}
