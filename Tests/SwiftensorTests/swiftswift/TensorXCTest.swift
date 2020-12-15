import XCTest
import Swiftensor

func XCTAssertEqual<T: Equatable>(_ expression1: Tensor<T>, _ expression2: Tensor<T>) {
    XCTAssertEqual(expression1.shape, expression2.shape)
    XCTAssertEqual(expression1.storage.data, expression2.storage.data)
}

func XCTAssertEqual<T: BinaryFloatingPoint>(_ expression1: Tensor<T>, _ expression2: Tensor<T>, accuracy: T = 1e-5) {
    
    XCTAssertEqual(expression1.shape, expression2.shape)
    
    for (a, b) in zip(expression1.storage.data, expression2.storage.data) {
        XCTAssertEqual(a, b, accuracy: accuracy)
    }
}