type RPCKind<K extends string, T extends {[name in K]: {
	request: any,
	response: any
}}> = T;
type TestRPC = RPCKind<"hello" | "bye", {
	hello: {
		request: string,
		response: string
	},
	bye: {
		request: number,
		response: string
	}
}>;
const Hello: () => Handler<keyof TestRPC, TestRPC> = () => ({
	hello: (request: string) => Promise.resolve({ response: `hello ${request}` }),
	bye: (req: number) => Promise.resolve({ response: `bye x${req}` })
});
const HelloKeys: (keyof TestRPC)[] = ["hello", "bye"];

type Handler<K extends string, RPC extends RPCKind<K, any>> = {
	[k in K]: (request: RPC[k]["request"]) => Promise<{ response: RPC[k]["response"], transferables?: any[] }>
}
/*type RpcRequest<Methods extends RPCKind<any, any>> = {
	method: keyof Methods,
	data: Methods[keyof Methods]["request"]
}*/
type RpcRequest<K extends string, RPC extends RPCKind<K, any>> = {
	[k in K]: {
		method: k,
		data: RPC[k]["request"]
	}
}[K];

type RpcResponse<K extends string, RPC extends RPCKind<K, any>> = {
	error: false,
	response: RPC[keyof RPC]["response"]
} | {
		error: true,
		response: any
	}
export function createRPCWorker<K extends string, RPC extends RPCKind<K, any>>(handler: () => Handler<K, RPC>, methods: K[]) {
	const workerCode = (getHandler: () => Handler<K, RPC>) => {
		const handler = getHandler();
		self.addEventListener('message', e => {
			const data: RpcRequest<K, RPC> = e.data;
			handler[data.method](data.data).then(({ response, transferables }) => {
				self.postMessage({ error: false, response } as RpcResponse<K, RPC>, transferables as any);
			}).catch(e => {
				(self as any).postMessage({ error: true, response: e } as RpcResponse<K, RPC>);
			});
		});
	}
	const worker = createWorker(workerCode, handler as any);
	const obj: {[meth in K]: (request: RPC[meth]["request"]) => Promise<RPC[meth]["response"]>} = {} as any;
	for (const method of methods) {
		obj[method] = data => {
			return new Promise<RPC[typeof method]["response"]>((resolve, reject) => {
				worker.postMessage({ method, data } as RpcRequest<K, RPC>);
				worker.addEventListener('message', (e: MessageEvent) => {
					const data = e.data as RpcResponse<K, RPC>;
					if (data.error) reject(data.response);
					else resolve(data.response);
				}, { once: true });
			});
		}
	}
	return obj;
}

function createWorker(workerFunc: (data: any) => void, data: string) {
	const src = `(${workerFunc})(${data});`;
	const blob = new Blob([src], { type: 'application/javascript' });
	const url = URL.createObjectURL(blob);
	return new Worker(url);
}


async function test() {
	console.log("testing");
	const worker = createRPCWorker(Hello, HelloKeys);
	console.log("abc", await worker.hello("abc"));
	console.log("def", await worker.hello("def"));
	console.log(123, await worker.bye(123));
}

test();