// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "eurus",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "eurus",
            targets: ["eurus"]),
        .library(
            name: "swiftensor",
            targets: ["swiftensor"]
        ),
        .executable(name: "examples", targets: ["examples"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git",from: "1.0.0")
    ],
    targets: [
        // Targets are the example building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "eurus",
            dependencies: ["swiftensor"],
            path: "./Sources/eurus"
        ),
        .target(
            name: "examples",
            dependencies: ["eurus","swiftensor"],
            path: "./Sources/examples"
        ),
        .target(
            name: "swiftensor",
            dependencies: [],
            path: "./Sources/swiftensor"
        ),
        .testTarget(
            name: "eurusTests",
            dependencies: ["eurus"]),
        .testTarget(
            name: "swiftensorTests",
            dependencies: ["swiftensor"]),
        
    ]
)
