#pragma once

#include <thread>
#include <queue>
#include <condition_variable>
#include <cassert>

struct AsyncFunction
{
	std::function<void()> fn;
	bool blocking;
	std::mutex lock;
	std::condition_variable cv;
};

class AsyncQueue {

protected:
	void emptyExecQueue()
	{
		std::lock_guard<std::mutex> queueLock(this->executionQueueLock);
		while (!executionQueue.empty())
		{
			auto exec = executionQueue.front();
			executionQueue.pop();

			if (exec->blocking)
			{
				std::lock_guard<std::mutex> lock(exec->lock);
				exec->fn();
				exec->cv.notify_all();
			}
			else
			{
				exec->fn();
				delete exec;
			}
		}
	}

public:
	void executeOnThread(std::function<void()> function, bool blocking = false)
	{

		AsyncFunction* fn = new AsyncFunction();
		fn->fn = function;
		fn->blocking = blocking;
		{
			std::lock_guard<std::mutex> lock(executionQueueLock);
			executionQueue.push(fn);
		}

		if(fn->blocking)
		{
			std::unique_lock<std::mutex> lock(fn->lock);
			fn->cv.wait(lock);
			lock.unlock();
			delete fn;
		}

	}

private:
	std::queue<AsyncFunction*> executionQueue;
	std::mutex executionQueueLock;
	std::condition_variable executionQueueConditionVariable;
};

	