import Foundation
import XCTest
@testable import Swiftensor

class TensorPerformanceTests: XCTestCase {
    
    func testTransposePerformance() {
        let a = Tensor<Int>.zeros([10, 10, 10, 10])
        measure {
            _ = a.transposed()
        }
    }
    
    func testSubscriptSubarrayPerformance() {
        let a = Tensor<Int>.zeros([1000, 1000])
        measure {
            _ = a[0..<300, 100..<200]
        }
    }
    
    func testStackPerformance0() {
        let a = Tensor<Int>.zeros([1000, 1000])
        measure {
            _ = Tensor.concatenate([a, a, a], along: 0)
        }
    }
    
    func testStackPerformance1() {
        let a = Tensor<Int>.zeros([1000, 1000])
        measure {
            _ = Tensor.concatenate([a, a, a], along: 1)
        }
    }
    
    func testDividePerformance() {
        let a = Tensor<Double>.linspace(low: 1, high: 1e7, count: 100000)
        let b = Tensor<Double>.linspace(low: 10, high: 1e7, count: 100000)
        measure {
            _ = divide(a, b)
        }
    }
    
    func testSqrtPerformance() {
        let a = Tensor<Double>.linspace(low: -10 * .pi, high: 10 * .pi, count: 1000000)
        measure {
            _ = _sqrt(a)
        }
    }
    
    func testSumPerformance0() {
        let a = Tensor<Double>.linspace(low: 0, high: 1e4, count: 100000)
        measure {
            _ = _sum(a)
        }
    }
    
    func testSumPerformance1() {
        let a = Tensor<Double>.linspace(low: 0, high: 1e4, count: 100000).reshaped([10, 10, 10, 10, 10])
        measure {
            _ = _sum(a, along: 2)
        }
    }
    
    func testNormalPerformance() {
        measure {
            _ = Tensor<Double>.normal(mu: 0, sigma: 1, shape: [100, 100])
        }
    }
    
    static var allTests: [(String, (TensorPerformanceTests) -> () throws -> Void)] {
        return [
            ("testTransposePerformance", testTransposePerformance),
            ("testSubscriptSubarrayPerformance", testSubscriptSubarrayPerformance),
            ("testStackPerformance0", testStackPerformance0),
            ("testStackPerformance1", testStackPerformance1),
            ("testDividePerformance", testDividePerformance),
            ("testSqrtPerformance", testSqrtPerformance),
            ("testSumPerformance0", testSumPerformance0),
            ("testSumPerformance1", testSumPerformance1),
            ("testNormalPerformance", testNormalPerformance)
        ]
    }
}
