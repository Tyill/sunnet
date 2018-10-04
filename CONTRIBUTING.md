# How to contribute

No external dependencies: boost, opencv, etc.

## Modules allowed for commits
 
 * [skynet](https://github.com/Tyill/skynet/tree/master/src/skynet) - only fix bug. Interface no changed
 * [snOperator](https://github.com/Tyill/skynet/tree/master/src/snOperator) - only new operator and fix bug
 * [snAuxFunc](https://github.com/Tyill/skynet/tree/master/src/snAux) - with the agreement
 * [python interface](https://github.com/Tyill/skynet/tree/master/python/libskynet) - no limits
 * [cpp interface](https://github.com/Tyill/skynet/tree/master/cpp) - without restrictions, but with the agreement
 * [c_sharp interface](https://github.com/Tyill/skynet/tree/master/c_sharp/libskynet) - no limits
 * [example](https://github.com/Tyill/skynet/tree/master/example) - no limits
 * [wiki](https://github.com/Tyill/skynet/wiki) - no limits, but reliably
 
## Modules forbidden for commits (only issue)

 * [snEngine](https://github.com/Tyill/skynet/tree/master/src/snEngine)
 * [snBase](https://github.com/Tyill/skynet/tree/master/src/snBase)
 
## Coding conventions

 * no any Template<> (only for native cpp)
 * camelCase for all names (except constants).
 

## Submitting changes

Please send a [GitHub Pull Request to skynet](https://github.com/Tyill/skynet/pull/new/master) 
with a clear list of what you've done (read more about [pull requests](https://help.github.com/articles/proposing-changes-to-your-work-with-pull-requests/). 


Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."
	
	