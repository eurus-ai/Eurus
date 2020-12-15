import XCTest
@testable import Swiftensor

class TensorUtilsTests: XCTestCase {
    func testBroadcastShape() {
        do {
            let x = [2,3,4]
            let y = [1,3,4]
            XCTAssertEqual(shapeForBroadcast(x,y),[2,3,4])
        }
        do {
            let x = [2,3,4]
            let y = [2,3]
            XCTAssertEqual(shapeForBroadcast(x,y),nil)
        }
        do {
            let x = [2,3,4]
            let y = [4]
            XCTAssertEqual(shapeForBroadcast(x,y),[2,3,4])
        }
        do {
            let x = [2,3,4]
            let y = [2,1,4]
            XCTAssertEqual(shapeForBroadcast(x,y),[2,3,4])
        }
    }

    static var allTests: [(String, (TensorUtilsTests) -> () throws -> Void)] {
        return [
            ("testBroadcastShape", testBroadcastShape)
        ]
    }
}
