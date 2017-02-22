const path = require('path');
const webpack = require('webpack');
const production = process.env.NODE_ENV == "production";
const ExtractTextPlugin = require("extract-text-webpack-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const plugins = [
	new ExtractTextPlugin("[name].[hash].css"),
	new webpack.DefinePlugin({
		
	}),
    new HtmlWebpackPlugin({
		title: 'Backchannel Survey',
		template: require('html-webpack-template'),
		inject: false,
		mobile: true,
		appMountId: 'app',
		excludeChunks: ['admin']
	}),
	new HtmlWebpackPlugin({
		title: 'Backchannel Survey',
		template: require('html-webpack-template'),
		inject: false,
		mobile: true,
		appMountId: 'app',
		filename: 'pcqxnugylresibwhwmzv/index.html',
		excludeChunks: ['client']
	})
];
if (production) {
	plugins.push(new webpack.optimize.UglifyJsPlugin({ compress: { warnings: false } }));
	plugins.push(new webpack.DefinePlugin({
		'process.env': {
			'NODE_ENV': `"production"`
		}
	}));
}
module.exports = {
	entry: {
		"client": "./client",
		"admin": "./admin"
	},
	devtool: 'source-map',
	output: {
		path: path.join(__dirname, "build"),
		filename: "[name].[hash].js",
		chunkFilename: "[id].js",
		publicPath: "/"
	},
	module: {
		loaders: [
			{
				test: /\.jsx?$/,
				exclude: /(node_modules|bower_components)/,
				loader: 'babel-loader',
				query: {
					presets: ['es2015', 'react'],
					plugins: ['transform-class-properties']
				}
			},
			{
				test: /\.tsx?$/,
				exclude: /(node_modules|bower_components)/,
				loader: 'awesome-typescript-loader'
			},
			{
				test: /\.css$/,
				loader: ExtractTextPlugin.extract('css-loader?sourceMap')
			},
			{
				test: /\.less$/,
				loader: ExtractTextPlugin.extract('css-loader?sourceMap!less-loader')
			},
			{ test: /\.woff(2)?(\?v=[0-9]\.[0-9]\.[0-9])?$/, loader: "url-loader?limit=10000&mimetype=application/font-woff&name=[name]-[hash].[ext]" },
			{ test: /\.(ttf|eot|svg|png|jpg|mp3)(\?v=[0-9]\.[0-9]\.[0-9])?$/, loader: "file-loader?name=[name]-[hash].[ext]" },
			{test: /\.jsx?$/, include: /react-pivot/, loader: 'babel-loader', query: {presets: ['es2015', 'react']} }
		],
	},
	resolve: {
		extensions: [".webpack.js", ".web.js", ".tsx", ".ts", ".jsx", ".js"]
	},
	plugins
};
