import XCTest
@testable import Swiftensor

class TensorRandomTests: XCTestCase {
    
    func testUniform() {
        let a = Tensor<Double>.uniform(low: -1, high: 1, shape: [100000])
        for e in a.storage.data {
            XCTAssert(-1 <= e && e < 1)
        }
    }
    
    func testNormal() {
        let a = Tensor<Double>.normal(mu: 0, sigma: 1, shape: [1000000])

        let mean = a.mean()
        XCTAssertEqual(mean, 0, accuracy: 1e-2)
        
        let std = (a*a).mean() - mean*mean
        XCTAssertEqual(std, 1, accuracy: 1e-2)
    }
    
    static var allTests: [(String, (TensorRandomTests) -> () throws -> Void)] {
        return [
            ("testUniform", testUniform),
            ("testNormal", testNormal)
        ]
    }
}
