const webpack = require('webpack');
const fs = require('fs');
const path = require('path');

class WatchDirPlugin {
    constructor(paths) {
        this.paths = paths.map(pa => path.resolve(pa));
    }
    apply(compiler) {
        compiler.plugin("this-compilation", (compilation, params) => {
            for(const pathname of this.paths) {
                const cont = fs.readdirSync(pathname);
                compiler.apply(new webpack.DefinePlugin({
                    "VERSIONS": JSON.stringify(cont)
                }));
            }
        });
        compiler.plugin("after-compile", (compilation, cb) => {
            compilation.contextDependencies.push(...this.paths);
            compilation.fileDependencies.push(...this.paths);
            cb();
        });
    }
};

module.exports = {
    entry: './plot.tsx',
    devtool: 'source-map',
    plugins: [
        new WatchDirPlugin([
            "../../trainNN/out"
        ]),
    ],
    output: {
        filename: 'dist/plot.js',
        publicPath: "evaluate/plot"
    },
    module: {
        rules: [
            { test: /\.tsx?$/, loader: 'awesome-typescript-loader' }
        ]
    }
}