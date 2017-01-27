const webpack = require('webpack');
const fs = require('fs');
const path = require('path');
const Html = require('html-webpack-plugin');
const ExtractTextPlugin = require("extract-text-webpack-plugin");

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
        /*compiler.plugin("after-compile", (compilation, cb) => {
            compilation.contextDependencies.push(...this.paths);
            compilation.fileDependencies.push(...this.paths);
            cb();
        });*/
    }
};

module.exports = {
    entry: './plot',
    devtool: 'source-map',
    plugins: [
        new WatchDirPlugin([
            "../../trainNN/out"
        ]),
        new Html({
            title: 'Backchanneler NN evaluation'
        }),
        new ExtractTextPlugin("styles.[hash].css")
    ],
    output: {
        path: path.resolve('./dist'),
        filename: 'plot.[hash].js',
        publicPath: "./"
    },
    resolve: {extensions: [".js", ".json", ".ts", ".tsx"]},
    module: {
        rules: [
            { test: /\.tsx?$/, loader: 'awesome-typescript-loader' },
            { test: /\.css$/, loader: ExtractTextPlugin.extract('css-loader') }
        ]
    }
}