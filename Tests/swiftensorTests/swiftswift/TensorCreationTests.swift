import XCTest
@testable import Swiftensor

class TensorCreationTests: XCTestCase {
    
    func testZeros() {
        XCTAssertEqual(Tensor<Int>.zeros([3, 4]),
                       Tensor(shape: [3, 4], elements: [Int](repeating: 0, count: 12)))
        
        XCTAssertEqual(Tensor<Float>.zeros([2, 5]),
                       Tensor(shape: [2, 5], elements: [Float](repeating: 0, count: 10)))
    }
    
    func testOnes() {
        XCTAssertEqual(Tensor<Int>.ones([3, 4]),
                       Tensor(shape: [3, 4], elements: [Int](repeating: 1, count: 12)))
        
        XCTAssertEqual(Tensor<Float>.ones([2, 5]),
                       Tensor(shape: [2, 5], elements: [Float](repeating: 1, count: 10)))
    }
    
    func testEye() {
        XCTAssertEqual(Tensor<Int>.eye(3),
                       Tensor(shape: [3, 3], elements: [1, 0, 0,
                                                         0, 1, 0,
                                                         0, 0, 1]))
        XCTAssertEqual(Tensor<Double>.eye(2),
                       Tensor(shape: [2, 2], elements: [1.0, 0.0,
                                                         0.0, 1.0]))
    }
    
    func testRange() {
        XCTAssertEqual(Tensor<Int>.range(from: 0, to: 5, stride: 1),
                       Tensor(shape: [5], elements: [0, 1, 2, 3, 4]))
        
        XCTAssertEqual(Tensor<Double>.range(from: -1, to: 5, stride: 1),
                                   Tensor<Double>(shape: [6], elements: [-1, 0, 1, 2, 3, 4]),
                                   accuracy: 1e-10)
        
        XCTAssertEqual(Tensor<Double>.range(from: -1, to: 2, stride: 0.5),
                                   Tensor<Double>(shape: [6], elements: [-1, -0.5, 0, 0.5, 1, 1.5]),
                                   accuracy: 1e-10)
    }
    
    func testLinspace() {
        XCTAssertEqual(Tensor<Float>.linspace(low: 0.1, high: 0.5, count: 9),
                                   Tensor<Float>(shape: [9],
                                                  elements: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
                                   accuracy: 1e-10)
        
        XCTAssertEqual(Tensor<Double>.linspace(low: 0.1, high: 1, count: 10),
                                   Tensor<Double>(shape: [10],
                                                   elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                                   accuracy: 1e-10)
    }
    
    static var allTests: [(String, (TensorCreationTests) -> () throws -> Void)] {
        return [
            ("testZeros", testZeros),
            ("testOnes", testOnes),
            ("testEye", testEye),
            ("testRange", testRange),
            ("testLinspace", testLinspace)
        ]
    }
}
