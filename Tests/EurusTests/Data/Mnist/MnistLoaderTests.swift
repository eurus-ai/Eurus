//
//  File.swift
//  
//
//  Created by howard on 2021/1/25.
//

import Foundation

import XCTest
import Foundation
@testable import Eurus

class MnistLoaderTests: XCTestCase {
    func testLoadMnist() {
        let ((images, labels), (imagesVal, labelsVal)) = MnistLoader.loadMnist(path: #file.replacingOccurrences(of: "MnistLoaderTests.swift", with: ""),type:Float.self)
        XCTAssertEqual(images.shape,[60_000,1,28,28])
        XCTAssertEqual(labels.shape,[60_000])
        XCTAssertEqual(imagesVal.shape,[10_000,1,28,28])
        XCTAssertEqual(labelsVal.shape,[10_000])
    }
    
    static var allTests: [(String, (MnistLoaderTests) -> () throws -> Void)] {
        return [
            ("testLoadMnist", testLoadMnist)
        ]
    }
}
