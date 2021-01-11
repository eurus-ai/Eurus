import XCTest

@testable import SwiftensorTests

XCTMain([
     testCase(SwiftensorTests.allTests),
     testCase(TensorArithmeticTests.allTests),
     testCase(TensorBoolTests.allTests),
     testCase(TensorConvTests.allTests),
     testCase(TensorCreationTests.allTests),
     testCase(TensorFloatingPointFunctionsTests.allTests),
     testCase(TensorPerformanceTests.allTests),
     testCase(TensorRandomTests.allTests),
     testCase(TensorReduceTests.allTests),
     testCase(TensorStackTests.allTests),
     testCase(TensorSubscriptTests.allTests),
     testCase(TensorTransformationTests.allTests)
])