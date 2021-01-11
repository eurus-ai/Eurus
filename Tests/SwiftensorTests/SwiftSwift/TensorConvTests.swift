import XCTest
import Foundation
@testable import Swiftensor

class TensorConvTests: XCTestCase {
    func testImg2Col() {
        let png = URL(fileURLWithPath: #file.replacingOccurrences(of: "TensorConvTests.swift", with: "Lenna.png"))
        let tensor = Tensor<UInt8>.fromImage(path: png.path)
        let shape = tensor!.shape
        let bachted = tensor!.reshaped([-1,shape[0],shape[1],shape[2]])
        let result = img2col(value: bachted, batchSize: 1, channels: 3, height: 512, width: 512, kernelHeight: 3, kernelWidth: 3, padding: 1, stride: 1)
        XCTAssertEqual(result.shape,[27,512*512])
    }
    
    static var allTests: [(String, (TensorConvTests) -> () throws -> Void)] {
        return [
            ("testImg2Col", testImg2Col)
        ]
    }
}
