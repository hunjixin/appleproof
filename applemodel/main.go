package main

import (
	"fmt"
	"io"
	"os"
	"path"
	"strings"
)

func main() {
	fmt.Println(move())
}

func move() error {
	srcPath := "/Users/waylon/code/fruits-360-100x100/Training"
	destPath := "./dataset"

	err := os.MkdirAll(path.Join(destPath, "apple"), 0777)
	if err != nil {
		return err
	}
	err = os.MkdirAll(path.Join(destPath, "not_apple"), 0777)
	if err != nil {
		return err
	}

	fruitsEntries, err := os.ReadDir(srcPath)
	if err != nil {
		return err
	}
	for _, fruitsEntry := range fruitsEntries {
		fruitName := fruitsEntry.Name()
		if strings.HasPrefix(fruitName, "Apple") {
			entries, err := os.ReadDir(srcPath + "/" + fruitName)
			if err != nil {
				return err
			}
			for _, entry := range entries {
				//todo copy entry
				srcFile := path.Join(srcPath, fruitName, entry.Name())
				destFile := path.Join(destPath, "apple", strings.ReplaceAll(fruitName, " ", "_")+"_"+entry.Name())
				srcFs, err := os.Open(srcFile)
				if err != nil {
					return err
				}
				dstFs, err := os.Create(destFile)
				if err != nil {
					return err
				}
				_, err = io.Copy(dstFs, srcFs)
				if err != nil {
					return err
				}
			}
		} else {
			entries, err := os.ReadDir(srcPath + "/" + fruitName)
			if err != nil {
				return err
			}
			for _, entry := range entries {
				//todo copy entry
				srcFile := path.Join(srcPath, fruitName, entry.Name())
				destFile := path.Join(destPath, "not_apple", strings.ReplaceAll(fruitName, " ", "_")+"_"+entry.Name())
				srcFs, err := os.Open(srcFile)
				if err != nil {
					return err
				}
				dstFs, err := os.Create(destFile)
				if err != nil {
					return err
				}
				_, err = io.Copy(dstFs, srcFs)
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}
