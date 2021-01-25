//
//  File.swift
//  
//
//  Created by howard on 2021/1/25.
//
import XCTest
import Foundation
@testable import Eurus

class LoaderTests: XCTestCase {
    func testLoadBinary() {
        let data = DataLoader.loadBinary(path: #file.replacingOccurrences(of: "LoaderTests.swift", with: "/Mnist/t10k-images.idx3-ubyte"))
        XCTAssertEqual(data.count,7840016)
    }
    
    static var allTests: [(String, (LoaderTests) -> () throws -> Void)] {
        return [
            ("testLoadBinary", testLoadBinary)
        ]
    }
}
