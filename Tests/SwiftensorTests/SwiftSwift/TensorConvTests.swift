import XCTest
import Foundation
@testable import Swiftensor

class TensorConvTests: XCTestCase {
    func testImg2ColInput() {
        let png = URL(fileURLWithPath: #file.replacingOccurrences(of: "TensorConvTests.swift", with: "Lenna.png"))
        let tensor = Tensor<UInt8>.fromImage(path: png.path)
        let shape = tensor!.shape
        let bachted = tensor!.reshaped([-1,shape[0],shape[1],shape[2]])
        let result = img2col(value: bachted, batchSize: 1, channels: 3, height: 512, width: 512, kernelHeight: 3, kernelWidth: 3, padding: 1, stride: 1)
        XCTAssertEqual(result.shape,[27,512*512])
    }
    
    func testImg2Col() {
        let tensor = Tensor<Int>(shape: [1,4,4], elements: Array(0..<16))
//        [[0,1,2,3],
//        [4,5,6,7],
//        [8,9,10,11],
//        [12,13,14,15]]
        let shape = tensor.shape
        let bachted = tensor.reshaped([-1,shape[0],shape[1],shape[2]])
        let result = img2col(value: bachted, batchSize: 1, channels: 1, height: 4, width: 4, kernelHeight: 3, kernelWidth: 3, padding: 0, stride: 1)
//        [[0,1,2,4,5,6,8,9,10],
//        [1,2,3,5,6,7,9,10,11],
//        [4,5,6,8,9,10,12,13,14],
//        [5,6,7,9,10,11,13,14,15]]
        
        let expectedArray = [[0,1,4,5],
                            [1,2,5,6],
                            [2,3,6,7],
                            [4,5,8,9],
                            [5,6,9,10],
                            [6,7,10,11],
                            [8,9,12,13],
                            [9,10,13,14],
                            [10,11,14,15]]
        let expected = Tensor<Int>(expectedArray)
        
        XCTAssertEqual(result,expected)
    }
    
    func testCol2Img() {
        let array = [[0,1,4,5],
                            [1,2,5,6],
                            [2,3,6,7],
                            [4,5,8,9],
                            [5,6,9,10],
                            [6,7,10,11],
                            [8,9,12,13],
                            [9,10,13,14],
                            [10,11,14,15]]
        let tensor = Tensor<Int>(array)
        let result = col2img(value: tensor, resultShape: [1,1,4,4], batchSize: 1, channels: 1, height: 4, width: 4, kernelHeight: 3, kernelWidth: 3, padding: 0, stride: 1)
        let expected = Tensor<Int>(shape: [1,1,4,4], elements: Array(0..<16))
        XCTAssertEqual(result,expected)
    }
    
    func testCol2ImgInput() {
        let png = URL(fileURLWithPath: #file.replacingOccurrences(of: "TensorConvTests.swift", with: "Lenna.png"))
        let tensor = Tensor<UInt8>.fromImage(path: png.path)
        let shape = tensor!.shape
        let bachted = tensor!.reshaped([-1,shape[0],shape[1],shape[2]])
        let result = img2col(value: bachted, batchSize: 1, channels: 3, height: 512, width: 512, kernelHeight: 3, kernelWidth: 3, padding: 1, stride: 1)
        let backward = col2img(value: result, resultShape: bachted.shape, batchSize: 1, channels: 3, height: 512, width: 512, kernelHeight: 3, kernelWidth: 3, padding: 1, stride: 1)
        XCTAssertEqual(bachted,backward)
    }
    
    static var allTests: [(String, (TensorConvTests) -> () throws -> Void)] {
        return [
            ("testImg2Col", testImg2Col),
            ("testCol2Img", testCol2Img),
            ("testImg2ColInput",testImg2ColInput),
            ("testCol2ImgInput",testCol2ImgInput)
        ]
    }
}
